# py-gboost

Python bindings for the [`gboost`](../README.md) gradient-boosting library.

It exposes a single `GradientBooster` class backed by the Rust core, with a
sklearn-like `fit(x, y)` / `predict(x)` API over NumPy `float32` arrays. All
training work (histogram binning, split search, per-batch update) runs in Rust;
`fit` and `fit_next` release the GIL while the booster grows.

## Installation

`py-gboost` is a [maturin](https://www.maturin.rs) package; it builds the Rust
extension and installs it as the importable module `gboostrs`.

```bash
# From a release build
pip install .

# Or, for in-place development
pip install maturin
maturin develop
```

Requires Python 3.9+. The import name is `gboostrs`:

```python
import gboostrs
gboostrs.__version__
```

## Quick start

```python
import numpy as np
from gboostrs import GradientBooster

rng = np.random.default_rng(1)
x = rng.standard_normal((100, 5)).astype(np.float32)  # (rows, cols)
y = rng.standard_normal(100).astype(np.float32)        # one value per row

# Regression
gb = GradientBooster(objective="squaredloss", iterations=100)
gb.fit(x, y)
scores = gb.predict(x)      # np.ndarray of float32, one score per row
gb.loss()                   # mean loss of the current predictions vs. the labels

# Binary classification (labels in {0, 1})
y = rng.integers(0, 2, 100).astype(np.float32)
gb = GradientBooster(objective="logloss", iterations=100)
gb.fit(x, y)
```

### Growing round by round

`fit(x, y)` trains up to `iterations` rounds and replaces any previously
trained model. `fit_next()` grows one additional round on the existing model
and returns `False` when a round produced no tree (no gain) — the natural
early-stopping signal.

```python
gb = GradientBooster(objective="squaredloss", iterations=0)
gb.fit(x, y)                # fit(0) builds the booster without growing trees

while gb.fit_next():
    print(gb.loss())
```

## `GradientBooster`

All parameters are keyword-only and have defaults, so `GradientBooster()` is a
valid, fully-configured instance.

| Parameter            | Default       | Description                                                                     |
| -------------------- | :-----------: | ------------------------------------------------------------------------------- |
| `iterations`         |     `100`     | Maximum boosting rounds run by `fit`.                                           |
| `objective`          | `"squaredloss"` | Loss to minimize: `"squaredloss"` (regression) or `"logloss"` (classification). |
| `base_score`         |     `0.0`     | Initial constant prediction before any tree is added.                            |
| `max_bins`           |     `255`     | Maximum number of quantization bins per column.                                  |
| `persist_path`       |   `None`      | Directory for on-disk artifacts, if persistence is used.                         |
| `parallel`           |   `True`      | Enable the parallel (Rayon) execution paths.                                     |
| `seed`               |      `0`      | Seed for reproducible random draws (subsampling).                                |
| `learning_rate`      |     `0.3`     | Per-tree update step size.                                                       |
| `subsample`          |     `1.0`     | Row subsampling fraction per tree.                                               |
| `colsample_bytree`   |     `1.0`     | Per-tree feature subsampling fraction.                                           |
| `colsample_bylevel`  |     `1.0`     | Per-level feature subsampling fraction.                                          |
| `colsample_bynode`   |     `1.0`     | Per-node feature subsampling fraction.                                           |
| `max_depth`          |      `6`      | Maximum tree depth (`0` = no limit).                                             |
| `max_leaves`         |      `0`      | Maximum number of leaves (`0` = no limit).                                       |
| `grow_policy`        | `"depthwise"` | Tree growth policy: `"depthwise"` or `"lossguide"`.                              |
| `reg_alpha`          |     `0.0`     | L1 regularizer on leaf weights.                                                  |
| `reg_lambda`         |     `1.0`     | L2 regularizer on leaf weights.                                                  |
| `min_split_gain`     |     `0.0`     | Minimum gain required to take a split.                                           |
| `min_child_weight`   |     `1.0`     | Minimum hessian sum allowed in a leaf.                                           |
| `allow_missing_splits` |  `False`   | Whether missing values may drive their own split.                                |
| `onehot_columns`     |     `[]`      | Indices of columns to treat as one-hot categoricals (exact-bin splits).         |

### Methods

- **`fit(x, y) -> None`** — Train the booster. `x` is a 2-D `float32` matrix
  of shape `(rows, cols)`; `y` is a 1-D `float32` target of length `rows`.
  Replaces any previously trained model.
- **`fit_next() -> bool`** — Grow one additional round on the trained model.
  Returns `False` when the round produced no tree (no gain).
- **`predict(x) -> np.ndarray`** — Score `x` (shape `(rows, cols)`, same
  column count as training) with the trained model; returns one `float32`
  score per row.
- **`loss() -> float | None`** — Mean loss of the current predictions against
  the training labels, or `None` before `fit`.
- **`get_params() -> dict`** — Return the constructor kwargs as a dict of
  `name -> value`; `GradientBooster(**gb.get_params())` reconstructs an
  equivalent instance.

## Notes

- **Zero-copy inputs.** The Rust core works on column-major (Fortran-order)
  arrays. Pass `x` as `np.asfortranarray(x)` to avoid a copy; row-major or
  strided inputs are transposed into column-major internally.
- **Error types.** I/O failures (e.g. an invalid `persist_path`) raise
  `IOError`; other problems (an untrained model, a `NaN` in the target, a
  column-count mismatch in `predict`, an invalid `objective`/`grow_policy`)
  raise `ValueError`.
- **`onehot_columns`.** Columns listed here are split on an exact bin (`==`)
  rather than an ordered threshold (`<=`), so high-cardinality categoricals
  are first-class. Other columns are treated as linear (ordered) features.
- **Determinism.** With a fixed `seed` and `subsample`, a run is reproducible.

## Tests

The test suite exercises the binding end-to-end (regression, classification,
`fit_next`, and error paths):

```bash
pip install pytest
pytest tests/
```
