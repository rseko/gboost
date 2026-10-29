//! One boosting iteration's data update for large, on-disk datasets: accumulate
//! the new tree's leaf values into the running predictions, then recompute the
//! per-row gradient/hessian pairs. The updated predictions are written back to
//! the `boost.<step>` file on disk.
//!
//! Functions:
//! - `update_large`: optional tree update plus gradient/hessian update,
//!   streaming `BATCH`-row windows from disk, optionally parallelized over
//!   batches.
//! - `sanitize_name`: maps a `Feature::name` to a file-system-safe extension.

use crate::dataset::{Dataset, IndexT, Objective, RowIndex};
use crate::tree::Tree;
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use zerocopy::{FromBytes, IntoBytes};

/// Base file names used for on-disk I/O. Full filenames are derived by
/// appending per-feature or per-step extensions (e.g. `feature.<name>`,
/// `target`, `boost.1`).
pub const FILENAME_INPUT: &str = "feature";
pub const FILENAME_TARGET: &str = "target";
pub const FILENAME_PREDICT: &str = "boost";

/// Sanitize special characters in a feature name.
pub fn sanitize_name(name: &str) -> String {
    name.replace(['.', '/', '\\'], "_")
}

/// Maximum number of rows processed in a single batch. The data is partitioned
/// into at most `BATCH` rows of each column, and `bits_per_batch` samples are
/// drawn from that batch for gradient computation.
pub const BATCH: usize = 65536;

/// Parameters controlling the incremental update pass.
#[derive(Clone, Debug)]
pub struct UpdateConfig {
    /// Previous boosting step, if any; its on-disk artifact is the base prediction.
    pub train_step: usize,
    /// Total number of rows in the dataset.
    pub total_rows: usize,
    /// Fallback prediction used when no previous step exists.
    pub initial: f32,
    /// Rows sampled per batch to compute loss derivatives (`0 < bits <= BATCH`).
    pub bits_per_batch: usize,
}

/// Run one boosting update pass over a large on-disk dataset.
///
/// For every full batch of `BATCH` rows the function reads the feature and
/// target columns (and the previous step's predictions, when present), adds the
/// candidate `tree`'s contribution to the prediction, then uses a bitmask to
/// sample `bits_per_batch` rows to recompute the loss gradient and hessian.
/// Those derivatives are written into `index` and the updated predictions
/// into `boost.<step>`. A final partial batch (the tail) is handled separately.
pub fn update_large<T, I, O>(
    data: &mut Dataset<T>,
    index: &mut [RowIndex<I, O>],
    tree: Option<&Tree>,
    path: &Path,
    config: &UpdateConfig,
    rng: &mut StdRng,
    parallel: bool,
) -> Result<()>
where
    T: Copy + Into<usize> + IntoBytes + FromBytes + Send + Sync,
    I: IndexT,
    O: Objective,
{
    assert_eq!(index.len(), data.rows());

    // The sampled rows per batch must scale with the dataset size, so the
    // in-memory `data(inputs)` view and the on-disk layout stay consistent.
    assert_eq!(config.bits_per_batch.clamp(1, BATCH), config.bits_per_batch);
    assert_eq!(
        config.total_rows / BATCH,
        data.rows() / config.bits_per_batch
    );
    // The in-memory tail (rows past the last full batch) must fit the sampled
    // tail, otherwise derivatives would be written out of range.
    let (tail, rest) = (
        config.total_rows % BATCH,
        data.rows() % config.bits_per_batch,
    );
    assert!(rest <= tail);

    let (rows, cols) = (data.rows(), data.columns());
    let names: Vec<String> = data.features().iter().map(|x| x.name().into()).collect();
    let batch_count = rows.div_ceil(config.bits_per_batch);
    let last_batch = rows - rest;

    // Reshape the in-memory data from per-column full-row vectors into per-batch
    // views: `input_slices[b][c]` is batch `b`'s slice of feature column `c`.
    let mut input_slices: Vec<Vec<&mut [T]>> = Vec::with_capacity(batch_count);
    for _ in 0..batch_count {
        input_slices.push(vec![]);
    }
    for items in data.slices_mut() {
        for (src, dst) in items
            .chunks_mut(config.bits_per_batch)
            .zip(input_slices.iter_mut())
        {
            dst.push(src);
        }
    }

    // Mirror the same per-batch split for the row-index slice, so each batch
    // writes its derivatives into the matching `index` window.
    let index_slices = index.chunks_mut(config.bits_per_batch).collect::<Vec<_>>();

    // Create and size the prediction file for this step up front so the
    // output is a fixed-length, fully written artifact for all rows.
    let suffix = (config.train_step + 1).to_string();
    let output = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path.join(FILENAME_PREDICT).with_extension(suffix))?;
    output.set_len((config.total_rows * std::mem::size_of::<f32>()) as u64)?;

    if parallel {
        // One deterministic seed per batch, so each worker re-derives its own
        // `StdRng` locally. With a fresh shuffle buffer per batch, each batch's
        // sampled rows are reproducible without shared mutable state.
        let seeds = rng
            .random_iter::<u64>()
            .take(batch_count)
            .collect::<Vec<_>>();

        // Each batch runs concurrently with its own file handles and buffers,
        // performing the same seek/read/process/write cycle as the sequential
        // path. The shuffle buffer is created per batch, not shared per thread.
        input_slices
            .into_par_iter()
            .zip(index_slices.into_par_iter())
            .zip(seeds.into_par_iter())
            .enumerate()
            .try_for_each_init(
                || {
                    // Open the dataset's file handles per worker (the `error`
                    // arm defers a failed open until the batch's turn).
                    let (files, error) = match UpdateFiles::open(path, &names, config.train_step) {
                        Ok(v) => (Some(v), None),
                        Err(e) => (None, Some(e)),
                    };
                    (files, error, UpdateBuffers::<T>::new(cols))
                },
                |(files, error, buffers), (idx, ((mut inputs, index), seed))| {
                    // Reconstruct this batch's RNG from its seed, then pick the
                    // sampled rows and the byte width to read (full batch, or
                    // the smaller tail for the last batch).
                    let mut rng = StdRng::seed_from_u64(seed);
                    let offset = idx * config.bits_per_batch;

                    let (mask, read_size) = if offset < last_batch {
                        let mut mask_gen = BitmaskGenerator::new(BATCH);
                        (mask_gen.next(&mut rng, config.bits_per_batch), BATCH)
                    } else {
                        buffers.predict.truncate(tail);
                        (BitmaskGenerator::new(tail).next(&mut rng, rest), tail)
                    };

                    if let Some(files) = files {
                        files.seek::<T>(idx * BATCH)?;
                        files.read(buffers, read_size, config.initial)?;
                        process_batch(buffers, &mut inputs, index, tree, &mask, offset);
                        files.output.write_all(buffers.predict.as_bytes())?;
                    } else if let Some(error) = error.take() {
                        return Err(error);
                    }
                    Ok(())
                },
            )
    } else {
        // Sequential path: one shared file-handle set, one shared scratch
        // workspace, and one shared bitmask generator, all reused across
        // batches (allocating exactly once up front).
        let mut files = UpdateFiles::open(path, &names, config.train_step)?;
        let mut buffers = UpdateBuffers::new(cols);
        let mut mask_gen = BitmaskGenerator::new(BATCH);

        // Walk the dataset in `bits_per_batch` global-row strides, pairing each
        // offset with the corresponding per-batch input/index slices.
        let counter = (0..rows).step_by(config.bits_per_batch);
        let iter = input_slices.into_iter().zip(index_slices);
        for (offset, (mut inputs, index)) in counter.zip(iter) {
            // Pick the sampled row mask and the byte width to read; on the
            // final (partial) batch also shrink the prediction buffer so the
            // tree-walk / gradient pass touches only the real rows.
            let (mask, read_size) = if offset < last_batch {
                (mask_gen.next(rng, config.bits_per_batch), BATCH)
            } else {
                buffers.predict.truncate(tail);
                (BitmaskGenerator::new(tail).next(rng, rest), tail)
            };

            // Read this batch's features + target + previous-step prediction
            // into `buffers`, fold in the candidate tree and recompute the
            // sampled rows' derivatives into `index`, then write the
            // updated predictions out at this batch's slot in `boost.<step>`.
            files.read(&mut buffers, read_size, config.initial)?;
            process_batch(&mut buffers, &mut inputs, index, tree, &mask, offset);
            files.output.write_all(buffers.predict.as_bytes())?;
        }
        Ok(())
    }
}

/// Add the candidate tree's contribution to every row's prediction, then for
/// the rows selected by `mask` compute their loss derivatives and record them
/// in `row_index` at the global position `offset + idx`.
fn process_batch<T: Copy + Into<usize>, I: IndexT, O: Objective>(
    buffers: &mut UpdateBuffers<T>,
    inputs: &mut [&mut [T]],
    index: &mut [RowIndex<I, O>],
    tree: Option<&Tree>,
    mask: &[u64],
    offset: usize,
) {
    // Add the candidate tree's leaf contribution to each row's running
    // prediction. `get_bin` maps a row and column to its bin id so the tree
    // can be traversed row by row.
    if let Some(tree) = tree {
        for (idx, y_pred) in buffers.predict.iter_mut().enumerate() {
            let get_bin = |col: usize| buffers.input[col * BATCH + idx].into();
            *y_pred += tree.find_leaf(get_bin).get_value();
        }
    }

    // Expand `mask` (one u64 word per 64 rows) into the flat set of row bit
    // positions that were sampled. `idx` is the sampled row's index within the
    // batch; `bit` is its position within this batch.
    let iter = mask.iter().enumerate().flat_map(|(i, n)| {
        (0..64)
            .filter(|j| *n & (1 << j) != 0)
            .map(move |j| i * 64 + j)
    });
    iter.enumerate().for_each(|(idx, bit)| {
        // Copy the sampled row's feature values from the batch buffer into the
        // in-memory inputs column view.
        for (src, dst) in buffers.input.chunks(BATCH).zip(inputs.iter_mut()) {
            dst[idx] = src[bit];
        }
        // Recompute the loss derivatives for this row using its true target and
        // the (possibly tree-adjusted) prediction, then store them.
        let deriv = O::derive(buffers.target[bit], buffers.predict[bit]);
        index[idx] = RowIndex::new(offset + idx, deriv);
    });
}

/// Per-batch scratch workspace for one `BATCH`-row window of data. Reused
/// across batches so each worker/thread allocates exactly once.
struct UpdateBuffers<T> {
    /// All feature columns, column-major: `input[col * BATCH + row]`.
    input: Vec<T>,
    /// True target values for the current batch (`target[row]`).
    target: Vec<f32>,
    /// Running predictions for the current batch (`predict[row]`), seeded
    /// from the previous step or `initial`, then extended by the candidate
    /// tree before gradient computation.
    predict: Vec<f32>,
}

impl<T: Clone + FromBytes> UpdateBuffers<T> {
    /// Allocate column-major buffers for `columns` feature columns, sized for
    /// one `BATCH`-row window (the tail batch reuses the same capacity).
    fn new(columns: usize) -> Self {
        Self {
            input: vec![T::new_zeroed(); BATCH * columns],
            target: vec![0.0; BATCH],
            predict: vec![0.0; BATCH],
        }
    }
}

/// Open file handles for a single on-disk dataset, positioned at a given
/// offset. `predict` is the previous step's prediction (if any) and `output`
/// is the current step's prediction file being written.
struct UpdateFiles {
    /// One handle per feature column, column-major, in `features` order.
    input: Vec<File>,
    /// Target values file.
    target: File,
    /// Previous step's predictions; `None` on the first step.
    predict: Option<File>,
    /// Current step's predictions file, being written.
    output: File,
}

impl UpdateFiles {
    /// Open all on-disk handles for a dataset under `path`. Each feature is
    /// read from `feature.<name>` (special chars in the name are replaced with `_`),
    /// the target from `target`, the previous step's predictions from
    /// `boost.<train_step>` when present, and the current step's output file
    /// `boost.<train_step + 1>` (step 0 on the first pass).
    fn open(path: &Path, names: &[String], train_step: usize) -> Result<Self> {
        let input = names
            .iter()
            .map(|name| {
                let name = sanitize_name(name);
                OpenOptions::new()
                    .read(true)
                    .open(path.join(FILENAME_INPUT).with_extension(&name))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let target = OpenOptions::new()
            .read(true)
            .open(path.join(FILENAME_TARGET))?;

        let predict = if train_step > 0 {
            let suffix = train_step.to_string();
            Some(
                OpenOptions::new()
                    .read(true)
                    .open(path.join(FILENAME_PREDICT).with_extension(suffix))?,
            )
        } else {
            None
        };

        let suffix = (train_step + 1).to_string();
        let output = OpenOptions::new()
            .write(true)
            .open(path.join(FILENAME_PREDICT).with_extension(suffix))?;

        Ok(Self {
            input,
            target,
            predict,
            output,
        })
    }

    /// Position every handle at global row `offset`: feature files at
    /// `offset * size_of::<T>()`, and the target/predict/output files at
    /// `offset * size_of::<f32>()`, from the start of each file.
    fn seek<T: Sized>(&mut self, offset: usize) -> Result<()> {
        let position = (offset * std::mem::size_of::<T>()) as u64;
        for file in self.input.iter_mut() {
            file.seek(SeekFrom::Start(position))?;
        }

        let position = (offset * std::mem::size_of::<f32>()) as u64;
        self.target.seek(SeekFrom::Start(position))?;
        if let Some(file) = &mut self.predict {
            file.seek(SeekFrom::Start(position))?;
        }
        self.output.seek(SeekFrom::Start(position))?;
        Ok(())
    }

    /// Read `size` rows of feature columns, prediction, and target into the
    /// workspace from the current file positions (advancing each handle).
    /// When the previous step is absent, the prediction column is filled with
    /// `initial` instead of being read.
    fn read<T: IntoBytes + FromBytes>(
        &mut self,
        buffers: &mut UpdateBuffers<T>,
        size: usize,
        initial: f32,
    ) -> Result<()> {
        for (file, slice) in self.input.iter_mut().zip(buffers.input.chunks_mut(BATCH)) {
            file.read_exact(slice[..size].as_mut_bytes())?;
        }
        // Seed predictions from the previous step, or the constant initial value
        // on the first step.
        if let Some(file) = self.predict.as_mut() {
            file.read_exact(buffers.predict[..size].as_mut_bytes())?;
        } else {
            buffers.predict.fill(initial);
        }
        self.target
            .read_exact(buffers.target[..size].as_mut_bytes())?;
        Ok(())
    }
}

/// Generates pseudorandom bitmasks selecting a subset of rows within a batch.
/// Each call to [`BitmaskGenerator::next`] returns a uniformly random subset
/// of the requested size, independent of previous calls (rows may be
/// re-sampled across calls).
struct BitmaskGenerator(Vec<u16>);

impl BitmaskGenerator {
    fn new(total: usize) -> Self {
        assert!(total <= BATCH);
        Self((0..total).map(|i| i as u16).collect())
    }

    /// Return a bitmask of `bits` uniformly-randomly chosen rows out of
    /// `total`. Successive calls are independent: each one picks a fresh
    /// random subset of exactly `bits` rows (no replacement within a call).
    fn next(&mut self, rng: &mut StdRng, bits: usize) -> Vec<u64> {
        let total = self.0.len();
        assert!(bits <= total);

        if bits <= total / 2 {
            // Start from an empty mask and set the chosen bits.
            let mut mask = vec![0; total.div_ceil(64)];
            for bit in self.0.partial_shuffle(rng, bits).0.iter() {
                let (m, n) = (bit >> 6, bit & 63);
                mask[m as usize] |= 1 << n;
            }
            mask
        } else {
            // Chosen set is a majority: start all-ones and clear the unchosen.
            let mut mask = vec![u64::MAX; total.div_ceil(64)];
            for bit in self.0.partial_shuffle(rng, total - bits).0.iter() {
                let (m, n) = (bit >> 6, bit & 63);
                mask[m as usize] &= !(1 << n);
            }
            // `u64::MAX` seed set every bit including the spurious ones past
            // `total`; mask the last word down to just the valid `last_bits`
            // so rows beyond the batch aren't marked.
            if !total.is_multiple_of(64) {
                let last_bits = total % 64;
                *mask.last_mut().unwrap() &= (1 << last_bits) - 1;
            }
            mask
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Feature, SquaredLoss};
    use crate::tree::{Node, NodeSplit};
    use rand::Rng;

    const NUM_ROWS: usize = 100_000;
    const NUM_SAMPLES: usize = 30_000;
    const TEST_SEED: u64 = 42;

    /// A random dataset with a two feature columns.
    fn build_data() -> Dataset<u8> {
        let mut rng = StdRng::seed_from_u64(TEST_SEED);
        let mut random_row = || {
            let mut data = vec![0; NUM_ROWS];
            rng.fill_bytes(data.as_mut_slice());
            data
        };

        Dataset::new(
            vec![
                Feature::linear(Some("1".into()), vec![]),
                Feature::linear(Some("2".into()), vec![]),
            ],
            vec![random_row(), random_row()],
            vec![1.0; NUM_ROWS],
        )
        .unwrap()
    }

    /// A simple tree that predicts a ±0.125 per row.
    fn build_tree() -> Tree {
        let root = Node::default_split(NodeSplit {
            feature_id: 0,
            bin_id: 127,
            is_range: true,
            with_missing: true,
            positive_id: 1,
            negative_id: 2,
            gain: 0.0,
        });
        let left = Node::default_leaf(-0.125);
        let right = Node::default_leaf(0.125);
        Tree::new(vec![root, left, right]).unwrap()
    }

    /// Write dataset to temp files.
    fn write_dataset(path: &Path, data: &Dataset<u8>) {
        for (feature, row) in data.features().iter().zip(data.inputs()) {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path.join(FILENAME_INPUT).with_extension(feature.name()))
                .unwrap();
            file.write_all(row.as_bytes()).unwrap();
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.join(FILENAME_TARGET))
            .unwrap();
        file.write_all(data.target().as_bytes()).unwrap();
    }

    #[test]
    fn large_test() {
        // Create the dataset and write it to disk.
        let (mut data, tree) = (build_data(), build_tree());
        let mut rng = StdRng::seed_from_u64(TEST_SEED);
        let temp_dir = tempfile::TempDir::new().unwrap();
        write_dataset(temp_dir.path(), &data);

        // Truncate the dataset.
        let sampled_rows = NUM_ROWS * NUM_SAMPLES / BATCH;
        let mut index = RowIndex::<u32, SquaredLoss>::new_array(sampled_rows);
        for row in data.inputs_mut() {
            row.truncate(sampled_rows);
        }

        // Run the first update.
        let mut config = UpdateConfig {
            train_step: 0,
            total_rows: NUM_ROWS,
            initial: 0.5,
            bits_per_batch: NUM_SAMPLES,
        };
        assert!(
            update_large(
                &mut data,
                &mut index,
                Some(&tree),
                temp_dir.path(),
                &config,
                &mut rng,
                /*parallel*/ false
            )
            .is_ok()
        );

        // Verify the resulting dataset.
        let sum_inputs = |a: &[u8]| a.iter().map(|x| *x as u64).sum::<u64>();
        assert_eq!(sum_inputs(&data.inputs()[0]), 5810726);
        assert_eq!(sum_inputs(&data.inputs()[1]), 5832598);
        assert_eq!(
            index.iter().map(|x| x.grad_hess().0).sum::<f32>(),
            -22910.25
        );

        // Run the second update.
        config.train_step += 1;
        assert!(
            update_large(
                &mut data,
                &mut index,
                Some(&tree),
                temp_dir.path(),
                &config,
                &mut rng,
                /*parallel*/ true
            )
            .is_ok()
        );

        // Verify the resulting dataset.
        assert_eq!(sum_inputs(&data.inputs()[0]), 5819437);
        assert_eq!(sum_inputs(&data.inputs()[1]), 5827921);
        assert_eq!(
            index.iter().map(|x| x.grad_hess().0).sum::<f32>(),
            -22950.50
        );

        // Verify the resulting file.
        let mut predict = vec![0.0_f32; NUM_ROWS];
        let mut file = OpenOptions::new()
            .read(true)
            .open(temp_dir.path().join(FILENAME_PREDICT).with_extension("2"))
            .unwrap();
        file.read_exact(predict.as_mut_bytes()).unwrap();

        assert_eq!(predict[NUM_ROWS - 3..], [0.25, 0.75, 0.25]);
        assert_eq!(predict.iter().sum::<f32>(), 49970.5);
    }

    #[test]
    fn missing_files() {
        // Verify that the update fails if the files are missing.
        let mut data = build_data();
        let mut index = RowIndex::<u32, SquaredLoss>::new_array(NUM_ROWS);
        let mut rng = StdRng::seed_from_u64(TEST_SEED);
        let temp_dir = tempfile::TempDir::new().unwrap();

        let config = UpdateConfig {
            train_step: 0,
            total_rows: NUM_ROWS,
            initial: 0.5,
            bits_per_batch: BATCH,
        };
        assert!(
            update_large(
                &mut data,
                &mut index,
                None,
                temp_dir.path(),
                &config,
                &mut rng,
                /*parallel*/ true
            )
            .is_err()
        );

        // Create the files and re-try (no tree).
        write_dataset(temp_dir.path(), &data);
        assert!(
            update_large(
                &mut data,
                &mut index,
                None,
                temp_dir.path(),
                &config,
                &mut rng,
                /*parallel*/ true
            )
            .is_ok()
        );
    }

    #[test]
    fn bit_generator() {
        // (total, bits) pairs spanning both branches and the branch boundary.
        // Branch A is `bits <= total / 2`; branch B is `bits > total / 2`.
        let cases: &[(usize, usize)] = &[
            (2, 0),             // branch A, none selected
            (2, 1),             // branch A, minimal
            (2, 2),             // branch B, all selected
            (10, 5),            // branch A, exactly total / 2
            (10, 6),            // branch B, just over total / 2
            (BATCH, BATCH / 2), // branch A, full BATCH
            (BATCH, BATCH - 1), // branch B, near BATCH, non-multiple of 64
            (BATCH, BATCH),     // branch B, full BATCH, all selected
        ];

        /// Popcount of the mask, summed across its `u64` words.
        fn mask_popcount(mask: &[u64]) -> u64 {
            mask.iter().map(|w| w.count_ones() as u64).sum()
        }

        for &(total, bits) in cases {
            let mut rng = StdRng::seed_from_u64(TEST_SEED);
            let mut mask_gen = BitmaskGenerator::new(total);
            let mask = mask_gen.next(&mut rng, bits);
            assert_eq!(mask_popcount(&mask), bits as u64);
        }
    }
}
