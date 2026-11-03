//! Booster orchestration: growing boosting rounds, persisting model
//! artifacts, and re-initializing a booster from saved state.
//!
//! Structures:
//! - `BoostModel`: one booster - `fit_next` grows and applies one tree;
//!   `init_small` / `init_large` build a booster from in-memory data or an
//!   on-disk dataset.
//! - `ModelInit`: streams feature/target columns to disk for a large
//!   dataset while recording the feature header lines.

use crate::boost_tree::TreeBuilder;
use crate::dataset::{Dataset, Feature, FeatureKind, IndexT, Objective, RowIndex};
use crate::options::Options;
use crate::parser::{parse_model, parse_trees};
use crate::quantize::Quantizer;
use crate::tree::{NodeKind, Tree};
use crate::update_large::{
    BATCH, FILENAME_INPUT, FILENAME_TARGET, UpdateConfig, sanitize_name, update_large,
};
use crate::update_small::update_small;

use itertools::Itertools;
use num_traits::{Float, PrimInt};
use rand::{SeedableRng, rngs::StdRng};
use std::fs::{File, OpenOptions};
use std::io::{Read, Result, Write};
use std::path::PathBuf;
use thiserror::Error;
use zerocopy::{FromBytes, Immutable, IntoBytes};

/// The persisted booster artifact: one feature header line per column, a
/// blank line, then one tree per boosting round separated by blank lines.
pub const FILENAME_MODEL: &str = "model.txt";

/// Errors raised by a booster during training or (re)initialization.
#[derive(Error, Debug)]
pub enum BoostError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The persisted model text failed to parse.
    #[error("parser error @{0}")]
    Parser(String),
    /// The training data failed a `Dataset` consistency check.
    #[error("dataset error: {0}")]
    Dataset(String),
    /// A large-dataset operation was requested without a persistence path.
    #[error("missing persistence path")]
    MissingPath,
}

/// A `Result` with the booster-specific error type.
pub type BoostResult<T> = std::result::Result<T, BoostError>;

/// One gradient-boosting learner: it owns the (possibly offloaded) training
/// data, the per-row gradient/hessian index, the running predictions, and
/// the trees grown so far.
pub struct BoostModel<T, I, O> {
    options: Options,
    data: Dataset<T>,
    index: Vec<RowIndex<I, O>>,
    predict: Vec<f32>,
    model: Vec<Tree>,
    offload: Option<UpdateConfig>,
}

impl<T, I, O> BoostModel<T, I, O>
where
    T: PrimInt + Copy + Into<usize> + IntoBytes + FromBytes + Send + Sync,
    I: IndexT,
    O: Objective,
{
    /// Grow one boosting tree over the current gradient/hessian index and
    /// fold it into the running predictions.
    ///
    /// The tree is grown on a freshly seeded RNG (the global seed plus the
    /// current tree count, so each round is reproducible independently of
    /// how many rounds preceded it). A tree whose root is a leaf contributes
    /// nothing, so the round is skipped and `false` is returned; otherwise
    /// the tree is applied to the data, appended to the model file (when a
    /// persistence path is set), and `true` is returned.
    pub fn fit_next(&mut self) -> Result<bool> {
        // Per-round RNG: global seed + round index, so each round is
        // independently reproducible.
        let parallel = self.options.boost_options.parallel;
        let seed = self.options.boost_options.seed + self.model.len() as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        // Grow the tree and bail early if the root is a leaf (no gain).
        let builder = TreeBuilder::new(
            &mut self.index,
            &self.data,
            &self.options.tree_options,
            &self.options.node_options,
            &mut rng,
            parallel,
        );
        let tree = builder.execute();
        if let NodeKind::Leaf(_) = tree.get(0).kind {
            return Ok(false);
        }

        // Apply the tree: offload path streams over disk, small path folds
        // in-memory.
        let path = self.options.boost_options.persist_path.as_ref();
        if let Some(config) = &mut self.offload {
            config.train_step += 1;
            rng = StdRng::seed_from_u64(seed + 1);

            update_large(
                &mut self.data,
                &mut self.index,
                Some(&tree),
                path.unwrap(),
                config,
                &mut rng,
                parallel,
            )?;
        } else {
            update_small(
                &mut self.predict,
                &mut self.index,
                &self.data,
                Some(&tree),
                parallel,
            );
        }

        // Append the tree to the model file (one tree per round).
        if let Some(path) = path {
            let mut model_file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(path.join(FILENAME_MODEL))?;

            let content = format!("\n{}\n", tree.to_string(Some(self.data.features())));
            model_file.write_all(content.as_bytes())?;
        }

        // Record the tree and signal a successful round.
        self.model.push(tree);
        Ok(true)
    }

    /// Build a booster over in-memory training columns (the small-dataset
    /// path; no persistence required).
    ///
    /// The raw `inputs` columns are quantized to bin indices and the fitted
    /// boundaries are stored on the `features` descriptors (any provided
    /// names/kinds are kept, generated when absent). The running prediction
    /// starts at `base_score` for every row, the per-row index is seeded by
    /// one tree-less update, and any persisted model file is replayed so a
    /// saved booster can be reinitialized and trained further.
    pub fn init_small<F>(
        options: Options,
        inputs: &[&[F]],
        target: &[f32],
        features: Option<Vec<Feature>>,
    ) -> BoostResult<Self>
    where
        F: Float + FromBytes + Immutable + IntoBytes + Send + Sync,
    {
        // Fit one quantizer per column: the quantized bin indices become the
        // dataset inputs and the fitted boundaries replace the feature bins.
        let parallel = options.boost_options.parallel;
        let (all_rows, all_bins) =
            Quantizer::quantize_all(inputs, options.boost_options.max_bins as usize, parallel);

        // Without caller-supplied descriptors, synthesize linear features
        // named by their column position.
        let mut features = features.unwrap_or_else(|| {
            (0..inputs.len())
                .map(|i| Feature::linear(Some(i.to_string()), vec![]))
                .collect()
        });

        // Keep each feature's name and kind, but store the fitted boundaries.
        for (feat, bins) in features.iter_mut().zip(all_bins) {
            let name = (!feat.name().is_empty()).then(|| feat.name().to_string());
            let bins = bins
                .into_iter()
                .map(|x| x.to_f32().unwrap_or(f32::NAN))
                .collect();
            *feat = Feature::new(name, feat.kind(), bins);
        }

        // Assemble the dataset from the quantized columns and targets.
        let data = Dataset::new(features, all_rows, target.to_vec())
            .map_err(|err| BoostError::Dataset(err.to_string()))?;

        // Seed the running prediction and the per-row gradient/hessian index
        // from the base score (no tree yet).
        let mut index = RowIndex::<I, O>::new_array(target.len());
        let mut predict = vec![options.boost_options.base_score; target.len()];
        update_small(&mut predict, &mut index, &data, None, parallel);

        // Replay any saved model: the leading blank line (and the feature
        // header block, already described by `data.features()`) is skipped
        // so the parse starts at the first tree and keeps consistent line
        // numbers. Each replayed tree folds into `predict` in turn.
        let mut model = vec![];
        if let Some(path) = options.boost_options.persist_path.as_ref()
            && let Ok(mut model_file) = File::open(path.join(FILENAME_MODEL))
        {
            let mut content = String::new();
            model_file.read_to_string(&mut content)?;
            let start = content.chars().position(|c| c != '\n').unwrap_or(0);

            // `start + 1` is the 1-based line number of the first tree,
            // so parse errors point at the right source line.
            model = parse_trees(&content[start..], data.features(), start + 1)
                .map_err(|err| BoostError::Parser(format!("{err:#}")))?;
            for tree in model.iter() {
                update_small(&mut predict, &mut index, &data, Some(tree), parallel);
            }
        }

        Ok(Self {
            options,
            data,
            index,
            predict,
            model,
            offload: None,
        })
    }

    /// Build a booster over an on-disk dataset (the large/offload path).
    ///
    /// The feature descriptors and saved trees come from the model file; the
    /// target file sizes the dataset. Rows are not loaded wholesale - an
    /// in-memory view holds one row per `bits_per_batch` sample (the
    /// `subsample` ratio is baked into the sample size and reset to `1.0`),
    /// and the first `update_large` pass (no tree) streams the full dataset
    /// to seed the per-row index from the base score. `offload` is set so
    /// later `fit_next` rounds take the streaming `update_large` path.
    pub fn init_large(mut options: Options) -> BoostResult<Self> {
        // The persistence path is mandatory for the offload path.
        let path = match options.boost_options.persist_path.as_ref() {
            Some(p) => p,
            None => return BoostResult::Err(BoostError::MissingPath),
        };

        // Parse the feature headers and all previously saved trees.
        let mut model_file = File::open(path.join(FILENAME_MODEL))?;
        let mut content = String::new();
        model_file.read_to_string(&mut content)?;
        let (features, trees) =
            parse_model(&content).map_err(|err| BoostError::Parser(format!("{err:#}")))?;

        // The target file holds one f32 per dataset row; its length fixes the
        // dataset width without reading any row.
        let target_file = File::open(path.join(FILENAME_TARGET))?;
        let size_bytes = target_file.metadata()?.len() as usize;
        let rows = size_bytes / std::mem::size_of::<f32>();

        // The in-memory view holds `bits_per_batch` sampled rows per full
        // on-disk batch, so the `subsample` ratio is folded into the sample
        // size here and then reset: tree building sees an unsampled,
        // pre-sampled row set.
        let batch_size = (options.tree_options.subsample.clamp(0.0, 1.0) * BATCH as f64).round();
        let bits_per_batch = (batch_size as usize).max(1);
        let sample_rows = rows * bits_per_batch / BATCH;
        options.tree_options.subsample = 1.0; // reset the subsample

        // Zeroed placeholder inputs: real bin values are streamed into these
        // slices by `update_large`.
        let inputs = vec![vec![T::new_zeroed(); sample_rows]; features.len()];
        let mut data = Dataset::<T>::new(features, inputs, vec![])
            .map_err(|err| BoostError::Dataset(err.to_string()))?;
        let mut index = RowIndex::new_array(sample_rows);

        // Continue the saved booster: its `trees.len()` steps are done, so
        // the next step reads `boost.<len>` and writes `boost.<len + 1>`.
        let config = UpdateConfig {
            train_step: trees.len(),
            total_rows: rows,
            initial: options.boost_options.base_score,
            bits_per_batch,
        };

        let parallel = options.boost_options.parallel;
        let seed = options.boost_options.seed + trees.len() as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        // Seed pass (no tree): stream every batch, fold the base score into
        // the predictions, and write the sampled rows' derivatives to `index`.
        update_large(
            &mut data, &mut index, None, path, &config, &mut rng, parallel,
        )?;

        // No in-memory `predict` on this path - predictions stay on disk.
        Ok(Self {
            options,
            data,
            index,
            predict: vec![],
            model: trees,
            offload: Some(config),
        })
    }
}

/// Stages a large, on-disk dataset for offloaded training: feature columns
/// are quantized and written to `feature.<name>` files, the feature header
/// lines are streamed into the model file, and the finished descriptors can
/// be retrieved with [`Self::finish`].
pub struct ModelInit {
    path: PathBuf,
    model_file: File,
    max_bins: usize,
    features: Vec<Feature>,
}

impl ModelInit {
    /// Start staging a dataset under `path`, truncating any existing model
    /// file. `max_bins` caps the per-column bin count (`>= 1`).
    pub fn new(path: PathBuf, max_bins: usize) -> Result<Self> {
        let model_file = File::create(path.join(FILENAME_MODEL))?;
        Ok(Self {
            path,
            model_file,
            max_bins: max_bins.max(1),
            features: vec![],
        })
    }

    /// Stage one feature column: fit its quantizer, write the quantized bin
    /// indices to `feature.<name>`, append the feature's header line (name,
    /// kind, bin boundaries) to the model file, and record the descriptor.
    ///
    /// Returns the fitted bin boundaries.
    pub fn write_feature<F, T>(
        &mut self,
        name: &str,
        kind: FeatureKind,
        data: &[F],
    ) -> Result<Vec<F>>
    where
        F: Float + FromBytes + Immutable + IntoBytes + Send + Sync,
        T: PrimInt + Immutable + IntoBytes,
    {
        // Special characters in the name are mapped to `_` so the file
        // extension stays portable.
        let suffix = sanitize_name(name);
        let mut feature_file = File::create(self.path.join(FILENAME_INPUT).with_extension(suffix))?;

        // Fit the quantizer and write the quantized column to disk.
        let quantizer = Quantizer::fit_from(data, self.max_bins);
        let data_quant = quantizer.quantize::<T>(data);
        feature_file.write_all(data_quant.as_bytes())?;

        // Emit the header line: `name kind [b0, b1, ...]`.
        let bins = quantizer
            .bins
            .iter()
            .map(|x| x.to_f32().unwrap_or(f32::NAN))
            .collect::<Vec<_>>();
        let feature_line = format!(
            "{} {} [{}]\n",
            name,
            match kind {
                FeatureKind::Linear => "linear",
                FeatureKind::OneHot => "onehot",
            },
            bins.iter().join(", "),
        );
        self.model_file.write_all(feature_line.as_bytes())?;

        // Store the feature and return the thresholds.
        self.features
            .push(Feature::new(Some(name.into()), kind, bins));
        Ok(quantizer.bins)
    }

    /// Write the per-row target column (one f32 per row) to the dataset.
    pub fn write_target(&self, data: &[f32]) -> Result<()> {
        let mut target_file = File::create(self.path.join(FILENAME_TARGET))?;
        target_file.write_all(data.as_bytes())?;
        Ok(())
    }

    /// Consume the staged descriptors and return the feature list.
    pub fn finish(self) -> Vec<Feature> {
        self.features
    }
}
