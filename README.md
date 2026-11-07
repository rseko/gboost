# gboost

A gradient-boosting library in Rust.

The distinguishing feature of `gboost` is out-of-core learning: the dataset is
staged to disk and each boosting round streams it through the gradient/hessian
computation, keeping only a subsample of rows in RAM — not the whole table.
You can train on tens of billions of rows on a machine that could never hold
the full columnar data at once.

It also works in-memory (fully columnar, histogram-binned, optionally parallel
via `rayon`) for datasets that do fit in RAM, behind the same object-safe
[`Booster`] trait.

`gboost` follows the XGBoost model: the same histogram-based boosting
algorithm, the same gradient/hessian leaf-value math, and training parameters
that mirror XGBoost's names (`learning_rate`, `subsample`, `colsample_bytree`,
`max_depth`, `min_split_gain`, `reg_alpha` / `reg_lambda`, …). If you know
XGBoost, the options will feel familiar — the main differences are the Rust
columnar API, a couple of gboost-specific options, and the out-of-core path.

## Features

- **Out-of-core training** — stage columns to disk, then grow trees by
  streaming the dataset with only a subsample of rows held in RAM, so memory
  stays bounded regardless of dataset size. See [`ModelInit`] and
  `BoostModel::init_large`.
- **One-hot (categorical) features** — a column can split on an exact bin
  (`==`) rather than an ordered threshold (`<=`), so high-cardinality
  categoricals are a first-class feature kind, not just `Linear` splits.
- **Half-precision floats** — the feature type `F` is generic over any
  `QuantFloat`, so `f16` and `bf16` (from the `half` crate) are accepted in
  addition to `f32` input type.
- **Histogram binning** — values are quantized into a bounded number of bins
  (`max_bins` up to 65535), so split search scales with the bin count, not the
  raw data.
- **Parallel training** — split search, histogram building and the per-batch
  update pass run across threads via `rayon` (toggle with `set_parallel`).
- **Deterministic** — a single `seed` drives all random draws (subsampling),
  so runs are reproducible.

## Installation

```toml
[dependencies]
gboost = "0.1"
```

MIT-licensed. Edition 2024 requires Rust 1.85+.

### Crate features

| Feature       | Default | Description                                                        |
| ------------- | :-----: | ------------------------------------------------------------------ |
| `large-index` |   no    | Allow training targets with more than `u32::MAX` rows (`u64` row indices). |

## Training on data that doesn't fit in RAM

This is the primary use case. The flow is:

1. **Stage** the dataset to a persistence directory: for each feature column,
   fit its quantizer and write the bin indices to `feature.<name>`; write the
   target to `target`; the feature headers are appended to `model.txt`.
2. **Build** a booster over that on-disk dataset with `BoostModel::init_large`.
   Rows are *not* loaded wholesale — only a subsample of rows is kept in RAM,
   and the first pass streams the full dataset to seed the per-row
   gradient/hessian index.
3. **Fit** one or more rounds. Each `fit_next` round takes the streaming
   `update_large` path: it walks the on-disk batches, folds the new tree's
   leaf values into the running predictions, and recomputes the derivatives.

The persistence path is mandatory on this route — set it with
`Options::set_persist_path`.

```rust
use gboost::boost_model::{BoostModel, ModelInit};
use gboost::dataset::{SquaredLoss, FeatureKind};
use gboost::options::Options;
use std::path::PathBuf;

let persist_path = PathBuf::from("/tmp/gboost");
let options = Options::default()
    .set_persist_path(Some(persist_path.clone()))
    .set_max_bins(255)
    .set_subsample(0.5)
    .set_seed(42)
    .set_parallel(true);

// 1. Stage columns to disk. Each column is fit, quantized, and written.
let mut init = ModelInit::new(persist_path.clone(), 255)?;
for (name, column) in [
    ("age", &[/* one f32 per row */]),
    ("income", &[/* ... */]),
] {
    init.write_feature::<u8, _>(name, FeatureKind::Linear, column)?;
}
init.write_target(&target)?;          // one f32 per row
let _features = init.finish();        // get bin thresholds data

// 2. Build the booster over the on-disk dataset.
let mut model = BoostModel::<u8, u32, SquaredLoss>::init_large(options)?;

// 3. Grow rounds; each streams batches from disk.
model.fit_next()?;
model.fit_next()?;
```

> **Very wide datasets.** If the row count exceeds `u32::MAX`, enable the
> `large-index` cargo feature and instantiate with a `u64` index
> (`BoostModel::<u8, u64, SquaredLoss>`).
>
> **Many columns on multi-core machines.** In the parallel update path, each
> rayon worker holds one open file handle per feature column (plus the target
> and prediction files). With `T` threads and `C` columns this is roughly
> `T × (C + 2)` open descriptors at once, so a wide dataset on a many-core
> system can exhaust the process or system file-descriptor limit
> (`EMFILE`/`ENFILE`). Raise `ulimit -n`, or cap the worker count
> (e.g. via `RAYON_NUM_THREADS`) if you hit this.

## Training in-memory

For data that fits in RAM, use the `Booster` trait: build a boxed, object-safe
booster over in-memory feature columns, grow it, and predict. `inputs` holds
one column per feature, in feature order; `target` has one value per row.

```rust
use gboost::boost_api::{new_booster, Booster};
use gboost::options::{ObjectiveKind, Options};

let inputs: Vec<Vec<f32>> = vec![/* one column: one f32 per row */; /* n_features */];
let cols = inputs.iter().map(|c| c.as_slice()).collect::<Vec<_>>();
let target: Vec<f32> = vec![/* per-row */];

let options = Options::default()
    .set_objective(ObjectiveKind::SquaredLoss)
    .set_learning_rate(0.1);

let mut booster = new_booster::<f32>(options, &cols, &target)?;

// Grow up to 50 rounds; stops early if a round adds no tree.
booster.fit(50)?;

let predictions = booster.predict(&cols)?;
let per_row_loss = booster.loss();
```

### Growing round by round

`fit` is a convenience wrapper over `fit_next`. Each call to `fit_next`
grows one tree and folds it into the ensemble. To control the stopping
condition yourself, loop until it reports no gain:

```rust
use gboost::boost_api::Booster;

while booster.fit_next()? {
    // keep boosting
}
```

`fit_next` returns `Ok(false)` when a round produced no tree (no gain), which
is the natural early-stopping signal.

### Explicit feature descriptors

`new_booster` synthesizes `linear` features named by position. To supply your
own names and kinds, use `new_with_features`; bin boundaries are still refit
from the inputs. Each column is either `linear` (ordered threshold splits) or
`onehot` (exact-bin splits for categoricals).

```rust
use gboost::boost_api::{new_with_features, Booster};
use gboost::dataset::Feature;
use gboost::options::Options;

let inputs: Vec<Vec<f32>> = vec![/* one column per feature */];
let cols = inputs.iter().map(|c| c.as_slice()).collect::<Vec<_>>();
let target: Vec<f32> = vec![/* per-row */];

let options = Options::default().set_learning_rate(0.1);

let features = vec![
    Feature::linear("age", None),      // ordered threshold
    Feature::onehot("country", None),  // exact-bin categorical
];
let mut booster = new_with_features::<f32>(options, &cols, &target, features)?;
booster.fit(50)?;
```

## Options

`Options` aggregates booster-, tree-, and node-level settings. Parameter names
and defaults mirror XGBoost, plus a couple of gboost-specific options. Every
field is reachable through self-consuming `set_*` methods, so a fully
configured value is built in one chain from `Options::default()`:

```rust
let options = Options::default()
    .set_objective(ObjectiveKind::LogLoss)   // classification
    .set_base_score(0.5)
    .set_learning_rate(0.1)
    .set_max_depth(10)                       // 0 = no limit
    .set_max_leaves(100)                     // 0 = no limit
    .set_allow_missing_splits(true);
```

Options round-trip to a portable JSON config:

```rust
let json = options.to_json();          // defaults and `persist_path` are pruned
let restored = Options::from_json(json)?;
```

`persist_path` is machine-local and is never serialized; re-supply it via
`set_persist_path` after `from_json`.

## Python bindings

A Python package is provided under [`py-gboost/`](py-gboost/). See that
directory for its own setup and tests.

## License

MIT — see [LICENSE](LICENSE).
