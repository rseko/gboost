//! Booster orchestration: growing boosting rounds, persisting model
//! artifacts, and re-initializing a booster from saved state.
//!
//! Structures:
//! - `BoostModel`: one booster - `fit_next` grows and applies one tree;
//!   `init_small` / `init_large` build a booster from in-memory data or an
//!   on-disk dataset; `predict` scores raw input columns.
//! - `ModelInit`: streams feature/target columns to disk for a large
//!   dataset while recording the feature header lines.

use crate::boost_tree::TreeBuilder;
use crate::dataset::{Dataset, Feature, FeatureKind, IndexT, Objective, RowIndex};
use crate::options::Options;
use crate::parser::parse_model;
use crate::quantize::{QuantFloat, bin_all, bins_from, quantize, quantize_all};
use crate::tree::{NodeKind, Tree};
use crate::update_large::{
    BATCH, FILENAME_INPUT, FILENAME_TARGET, UpdateConfig, sanitize_name, update_large,
};
use crate::update_small::update_small;

use itertools::Itertools;
use num_traits::{Float, PrimInt};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{Result, Write};
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
    /// A `u64` row index was requested but the `large-index` feature is disabled.
    #[error("dataset size exceeds the {0} row limit")]
    DatasetTooLarge(usize),
}

/// A `Result` with the booster-specific error type.
pub type BoostResult<T> = std::result::Result<T, BoostError>;

/// An integer bin-index type: a primitive integer convertible to and from
/// `usize` and byte-serializable (`u8` for small bins, `u16` for large).
pub trait BinT: PrimInt + Copy + Into<usize> + IntoBytes + FromBytes + Send + Sync {}
impl<T: PrimInt + Copy + Into<usize> + IntoBytes + FromBytes + Send + Sync> BinT for T {}

/// One gradient-boosting learner: it owns the (possibly offloaded) training
/// data, the per-row gradient/hessian index, the running predictions, and
/// the trees grown so far.
pub struct BoostModel<T, I, O> {
    options: Options,
    data: Dataset<T>,
    index: Vec<RowIndex<I, O>>,
    predict: Vec<f32>,
    trees: Vec<Tree>,
    offload: Option<UpdateConfig>,
}

impl<T: BinT, I: IndexT, O: Objective> BoostModel<T, I, O> {
    /// Predict over a batch of raw (unquantized) feature columns.
    ///
    /// `inputs` holds one column per feature, in feature order; every
    /// column must have the same length. Each row's prediction is the
    /// configured base score plus the sum of the leaf values the row
    /// lands in across all trees. A value that fails to convert to `f32`
    /// is treated as NaN and routes through the missing-value branch.
    /// Parallelism follows the configured `parallel` option.
    pub fn predict<F>(&self, inputs: &[&[F]]) -> BoostResult<Vec<f32>>
    where
        F: Float + Sync,
    {
        // Verify the correctness of the inputs.
        let data_error = |s: &str| BoostError::Dataset(s.into());
        if inputs.len() != self.data.columns() {
            return Err(data_error("input columns do not match features"));
        }
        let rows = inputs[0].len();
        if inputs.iter().any(|col| col.len() != rows) {
            return Err(data_error("input matrix is not dense"));
        }

        // Compute one row's prediction: for each tree, resolve the row's bin
        // per column (`get_bin`) to walk to a leaf, then sum the leaf values.
        let predict = |row: usize| {
            let get_bin = |col: usize| {
                let value = inputs[col][row].to_f32().unwrap_or(f32::NAN);
                let bins = self.data.features()[col].bins();
                quantize::<f32, T>(&value, bins).into()
            };
            self.trees
                .iter()
                .map(|tree| tree.find_leaf(get_bin).get_value())
                .sum::<f32>()
        };

        // One slot per row, filled with `base_score + predict(row)`.
        let mut result = vec![0.0; rows];
        let base_score = self.options.boost_options.base_score;

        if self.options.boost_options.parallel {
            const BATCH_SIZE: usize = 1024;
            result
                .par_chunks_mut(BATCH_SIZE)
                .enumerate()
                .for_each(|(idx, chunk)| {
                    chunk.iter_mut().enumerate().for_each(|(i, dst)| {
                        *dst = base_score + predict(idx * BATCH_SIZE + i);
                    });
                });
        } else {
            for (idx, dst) in result.iter_mut().enumerate() {
                *dst = base_score + predict(idx);
            }
        }
        Ok(result)
    }

    /// Grow one boosting tree over the current gradient/hessian index and
    /// fold it into the running predictions.
    ///
    /// The tree is grown on a freshly seeded RNG (the global seed plus the
    /// current tree count, so each round is reproducible independently of
    /// how many rounds preceded it). A tree whose root is a leaf contributes
    /// nothing, so the round is skipped and `false` is returned; otherwise
    /// the tree is applied to the data, appended to the model file (when a
    /// persistence path is set), and `true` is returned.
    pub fn fit_next(&mut self) -> BoostResult<bool> {
        // Per-round RNG: global seed + round index, so each round is
        // independently reproducible.
        let parallel = self.options.boost_options.parallel;
        let seed = self.options.boost_options.seed + self.trees.len() as u64;
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
        let persist_path = self.options.boost_options.persist_path.as_ref();
        if let Some(config) = &mut self.offload {
            config.train_step += 1;
            rng = StdRng::seed_from_u64(seed + 1);

            update_large(
                &mut self.data,
                &mut self.index,
                Some(&tree),
                persist_path.unwrap(),
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
        if let Some(path) = persist_path {
            let mut model_file = OpenOptions::new()
                .append(true)
                .create(false)
                .open(path.join(FILENAME_MODEL))?;

            let content = format!(
                "{}{}\n",
                if !self.trees.is_empty() { "\n" } else { "" },
                tree.to_string(Some(self.data.features()))
            );
            model_file.write_all(content.as_bytes())?;
        }

        // Record the tree and signal a successful round.
        self.trees.push(tree);
        Ok(true)
    }

    /// Build a booster over in-memory training columns (the small-dataset
    /// path; no persistence required).
    ///
    /// Without a saved model file, the raw `inputs` columns are quantized to
    /// bin indices and the fitted boundaries (capped by `max_bins`) are
    /// stored on the `features` descriptors - any provided names/kinds are
    /// kept, generated when absent - and the header lines are written to the
    /// model file (when a persistence path is set).
    ///
    /// With a saved model file, the feature descriptors, stored boundaries,
    /// and trees are loaded from it and `inputs` are mapped against the
    /// stored boundaries, so a saved booster can be reinitialized and trained
    /// further with the same bin grid.
    pub fn init_small<F: QuantFloat>(
        options: Options,
        inputs: &[&[F]],
        target: &[f32],
        features: Option<Vec<Feature>>,
    ) -> BoostResult<Self> {
        let persist_path = options.boost_options.persist_path.as_ref();
        let parallel = options.boost_options.parallel;

        // Map the raw input columns to bin indices against one set of
        // boundaries (the stored ones when restoring, the fitted ones
        // otherwise).
        let quantize = |bins: &[Vec<F>]| {
            let pairs = inputs
                .iter()
                .zip(bins.iter())
                .map(|(a, b)| (*a, b.as_slice()))
                .collect::<Vec<_>>();
            quantize_all::<F, T>(&pairs, parallel)
        };

        // Restore a saved booster (model file present) or fit a fresh one.
        let (features, rows, trees) = if let Some(path) = persist_path
            && path.join(FILENAME_MODEL).exists()
        {
            let content = std::fs::read_to_string(path.join(FILENAME_MODEL))?;
            let (features, trees) =
                parse_model(&content).map_err(|err| BoostError::Parser(format!("{err:#}")))?;

            // The file stores boundaries as f32; convert them back to the
            // input type (a boundary `F` cannot represent maps to NaN).
            let convert = |x: &f32| F::from(*x).unwrap_or(F::nan());
            let all_bins = features
                .iter()
                .map(|feat| feat.bins().iter().map(convert).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            (features, quantize(&all_bins), trees)
        } else {
            // Without caller-supplied descriptors, synthesize linear features
            // named by their column position.
            let mut features = features.unwrap_or_else(|| {
                (0..inputs.len())
                    .map(|i| Feature::linear(Some(i.to_string()), vec![]))
                    .collect()
            });

            // Fit one quantizer per column and keep each feature's name and
            // kind, but store the fitted boundaries (as f32).
            let all_bins = bin_all(inputs, options.boost_options.max_bins as usize, parallel);
            for (feat, bins) in features.iter_mut().zip(all_bins.iter()) {
                let name = (!feat.name().is_empty()).then(|| feat.name().to_string());
                let bins = bins
                    .iter()
                    .map(|x| x.to_f32().unwrap_or(f32::NAN))
                    .collect();
                *feat = Feature::new(name, feat.kind(), bins);
            }

            // Start the model file with the feature header block so later
            // `fit_next` rounds can append their trees.
            if let Some(path) = persist_path {
                let mut model_file = File::create(path.join(FILENAME_MODEL))?;
                for feat in features.iter() {
                    model_file.write_all(format_feature(feat).as_bytes())?;
                }
                model_file.write_all(b"\n")?;
            }

            (features, quantize(&all_bins), vec![])
        };

        // Assemble the dataset from the quantized columns and targets, then
        // seed the running prediction and the per-row gradient/hessian index
        // from the base score (no tree yet), replaying each saved tree in turn.
        let data = Dataset::new(features, rows, target.to_vec())
            .map_err(|err| BoostError::Dataset(err.to_string()))?;
        let mut index = RowIndex::<I, O>::new_array(target.len());
        let mut predict = vec![options.boost_options.base_score; target.len()];

        update_small(&mut predict, &mut index, &data, None, parallel);
        for tree in &trees {
            update_small(&mut predict, &mut index, &data, Some(tree), parallel);
        }

        Ok(Self {
            options,
            data,
            index,
            predict,
            trees,
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
        let content = std::fs::read_to_string(path.join(FILENAME_MODEL))?;
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
        let batch_size = (options.tree_options.subsample.clamp(0.0, 1.0) * BATCH as f32).round();
        let bits_per_batch = (batch_size as usize).max(1);
        let sample_rows = (rows * bits_per_batch / BATCH).max(1);
        options.tree_options.subsample = 1.0; // reset the subsample

        // Zeroed placeholder inputs: real bin values are streamed into these
        // slices by `update_large`.
        let inputs = vec![vec![T::new_zeroed(); sample_rows]; features.len()];
        let mut data = Dataset::<T>::new(features, inputs, vec![]).unwrap();
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
            trees,
            offload: Some(config),
        })
    }
}

impl<T: Copy, I, O> std::fmt::Debug for BoostModel<T, I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let format_tree = |t: &Tree| t.to_string(Some(self.data.features()));
        let trees = self.trees.iter().map(format_tree).join("\n\n");
        write!(f, "{:?}\n{}", self.data, trees)
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
    pub fn write_feature<T, F>(
        &mut self,
        name: &str,
        kind: FeatureKind,
        data: &[F],
    ) -> Result<Vec<F>>
    where
        T: PrimInt + Immutable + IntoBytes,
        F: QuantFloat,
    {
        // Special characters in the name are mapped to `_` so the file
        // extension stays portable.
        let suffix = sanitize_name(name);
        let mut feature_file = File::create(self.path.join(FILENAME_INPUT).with_extension(suffix))?;

        // Fit the quantizer and write the quantized column to disk.
        let bins_quant = bins_from(data, self.max_bins);
        let data_quant = data
            .iter()
            .map(|x| quantize::<F, T>(x, &bins_quant))
            .collect::<Vec<_>>();
        feature_file.write_all(data_quant.as_bytes())?;

        // Emit the header line: `name kind [b0, b1, ...]`.
        let bins = bins_quant
            .iter()
            .map(|x| x.to_f32().unwrap_or(f32::NAN))
            .collect::<Vec<_>>();
        let feature = Feature::new(Some(name.into()), kind, bins);
        self.model_file
            .write_all(format_feature(&feature).as_bytes())?;

        // Store the feature and return the thresholds.
        self.features.push(feature);
        Ok(bins_quant)
    }

    /// Write the per-row target column (one f32 per row) to the dataset.
    pub fn write_target(&mut self, data: &[f32]) -> Result<()> {
        let mut target_file = File::create(self.path.join(FILENAME_TARGET))?;
        target_file.write_all(data.as_bytes())?;
        self.model_file.write_all(b"\n")?;
        Ok(())
    }

    /// Consume the staged descriptors and return the feature list.
    pub fn finish(self) -> Vec<Feature> {
        self.features
    }
}

/// Render one feature header line of the model file: `name kind [b0, ...]`.
fn format_feature(feat: &Feature) -> String {
    format!(
        "{} {} [{}]\n",
        feat.name().replace(' ', "_"),
        match feat.kind() {
            FeatureKind::Linear => "linear",
            FeatureKind::OneHot => "onehot",
        },
        feat.bins().iter().join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::SquaredLoss;
    use crate::update_large::FILENAME_PREDICT;
    use rand::{Rng, RngExt};
    use std::io::Read;

    const NUM_ROWS: usize = 1000;
    const NUM_COLS: usize = 10;

    fn build_data() -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(0);
        let mut data = vec![vec![0.0; NUM_ROWS]; NUM_COLS];
        for row in data.iter_mut() {
            rng.fill_bytes(row.as_mut_bytes());
        }
        let target = (0..NUM_ROWS).map(|_| rng.random_range(0.0..1.0)).collect();
        (data, target)
    }

    #[test]
    fn small_test() {
        let (data, target) = build_data();
        let inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();
        let options = Options::default();
        let mut model =
            BoostModel::<u8, u32, SquaredLoss>::init_small(options, &inputs, &target, None)
                .unwrap();
        model.fit_next().unwrap();
        model.fit_next().unwrap();

        assert_eq!(
            model.data.features()[0].bins()[192..196],
            [2.158268, 6.059814, 14.9356165, 28.31305]
        );
        assert_eq!(
            model.data.inputs()[0][..8],
            [46, 66, 113, 54, 45, 132, 83, 106]
        );
        assert_eq!(
            model.predict[..4],
            [0.24033014, 0.24033014, 0.20896423, 0.2427137]
        );
    }

    #[test]
    fn small_restore() {
        let (data, target) = build_data();
        let inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let options = Options::default()
            .set_subsample(0.5)
            .set_persist_path(Some(temp_dir.path().into()));

        // Run a single pass and create a second model.
        let mut model =
            BoostModel::<u8, u32, SquaredLoss>::init_small(options.clone(), &inputs, &target, None)
                .unwrap();
        model.fit_next().unwrap();

        let mut model_restored =
            BoostModel::<u8, u32, SquaredLoss>::init_small(options, &inputs, &target, None)
                .unwrap();

        // Run another pass for both models and compare.
        model.fit_next().unwrap();
        model_restored.fit_next().unwrap();
        assert_eq!(model.predict, model_restored.predict);
        assert_eq!(format!("{:?}", model), format!("{:?}", model_restored));
    }

    #[test]
    fn small_invalid() {
        let create = |options: Options, features: Option<Vec<Feature>>| {
            BoostModel::<u8, u32, SquaredLoss>::init_small::<f32>(options, &[&[]], &[], features)
        };
        let err1 = create(Options::default(), Some(vec![]));
        assert!(matches!(err1, Err(BoostError::Dataset(_))));

        let temp_dir = tempfile::TempDir::new().unwrap();
        let mut file = File::create(temp_dir.path().join(FILENAME_MODEL)).unwrap();
        file.write_all(b"error").unwrap();
        let options = Options::default().set_persist_path(Some(temp_dir.path().into()));
        let err2 = create(options, None);
        assert!(matches!(err2, Err(BoostError::Parser(_))));
    }

    #[test]
    fn large_test() {
        let (data, target) = build_data();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let options = Options::default()
            .set_subsample(0.5)
            .set_persist_path(Some(temp_dir.path().into()));

        // Create the model data on disk.
        let mut model_init = ModelInit::new(temp_dir.path().into(), 255).unwrap();
        for (idx, row) in data.iter().enumerate() {
            let kind = if idx % 2 == 0 {
                FeatureKind::Linear
            } else {
                FeatureKind::OneHot
            };
            model_init
                .write_feature::<u8, _>(&idx.to_string(), kind, row)
                .unwrap();
        }
        model_init.write_target(&target).unwrap();
        let features = model_init.finish();

        // Create the model and fit data.
        let mut model = BoostModel::<u8, u32, SquaredLoss>::init_large(options).unwrap();
        model.fit_next().unwrap();
        model.fit_next().unwrap();

        assert_eq!(model.data.features(), features);
        assert_eq!(
            model.offload,
            Some(UpdateConfig {
                train_step: 2,
                total_rows: NUM_ROWS,
                initial: 0.0,
                bits_per_batch: BATCH / 2,
            })
        );

        // Verify checksums of the predictions.
        let checksum = |idx: usize| -> u64 {
            let path = temp_dir
                .path()
                .join(FILENAME_PREDICT)
                .with_extension(idx.to_string());
            let mut file = File::open(path).unwrap();
            let mut buffer = vec![0_u8; NUM_ROWS * 4];
            file.read_exact(&mut buffer).unwrap();
            let chunks = buffer.as_bytes().as_chunks::<8>().0;
            chunks
                .iter()
                .fold(0_u64, |acc, part| acc ^ u64::from_le_bytes(*part))
        };
        assert_eq!(checksum(1), 387_416_207_820_305_698);
        assert_eq!(checksum(2), 12_201_181_808_612_248);
    }

    #[test]
    fn large_invalid() {
        let create = |options: Options| BoostModel::<u8, u32, SquaredLoss>::init_large(options);
        let err1 = create(Options::default());
        assert!(matches!(err1, Err(BoostError::MissingPath)));

        let temp_dir = tempfile::TempDir::new().unwrap();
        File::create(temp_dir.path().join(FILENAME_MODEL)).unwrap();
        let options = Options::default().set_persist_path(Some(temp_dir.path().into()));
        let err2 = create(options.clone());
        assert!(matches!(err2, Err(BoostError::Parser(_))));
    }

    #[test]
    fn single_row() {
        let options = Options::default();
        let mut model =
            BoostModel::<u8, u32, SquaredLoss>::init_small(options, &[&[1.0]], &[1.0], None)
                .unwrap();
        assert_eq!(model.fit_next().ok(), Some(false));
    }

    #[test]
    fn predict_test() {
        let (data, target) = build_data();
        let inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

        for parallel in [false, true] {
            let options = Options::default()
                .set_parallel(parallel)
                .set_base_score(0.5)
                .set_max_depth(3);
            let mut model =
                BoostModel::<u8, u32, SquaredLoss>::init_small(options, &inputs, &target, None)
                    .unwrap();
            for _ in 0..10 {
                assert_eq!(model.fit_next().ok(), Some(true));
            }

            let predict = model.predict(&inputs).unwrap();
            let mean_error = predict
                .iter()
                .zip(&target)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / predict.len() as f32;
            assert_eq!(mean_error, 0.22513878);
        }
    }

    #[test]
    fn predict_invalid() {
        let (data, target) = build_data();
        let mut inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();
        let options = Options::default();
        let model = BoostModel::<u8, u32, SquaredLoss>::init_small(options, &inputs, &target, None)
            .unwrap();

        let err1 = model.predict::<f32>(&[]);
        assert!(matches!(err1, Err(BoostError::Dataset(_))));
        let _ = inputs[1].split_off(..1);
        let err2 = model.predict::<f32>(&inputs);
        assert!(matches!(err2, Err(BoostError::Dataset(_))));
    }
}
