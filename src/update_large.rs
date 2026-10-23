//! One boosting iteration's data update for large, on-disk datasets: accumulate
//! the new tree's leaf values into the running predictions, then recompute the
//! per-row gradient/hessian pairs. The updated predictions are written back to
//! the `boost.<step>` file on disk.
//!
//! Functions:
//! - `update_large`: optional tree update plus gradient/hessian update,
//!   streaming `BATCH`-row windows from disk, optionally parallelized over
//!   batches.

use crate::dataset::{Dataset, Feature, Objective, RowIndex, RowIndexT};
use crate::tree::Tree;
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use zerocopy::{FromBytes, IntoBytes};

/// Base file names used for on-disk I/O. Full filenames are derived by
/// appending per-feature or per-step extensions (e.g. `feature.<name>`,
/// `target`, `boost.1`).
const FILENAME_INPUT: &str = "feature";
const FILENAME_TARGET: &str = "target";
const FILENAME_PREDICT: &str = "boost";

/// Maximum number of rows processed in a single batch. The data is partitioned
/// into at most `BATCH` rows of each column, and `bits_per_batch` samples are
/// drawn from that batch for gradient computation.
const BATCH: usize = 65536;

/// Parameters controlling the incremental update pass.
#[derive(Clone, Debug)]
pub struct UpdateConfig {
    /// Previous boosting step, if any; its on-disk artifact is the base prediction.
    pub train_step: Option<usize>,
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
/// Those derivatives are written into `data.index` and the updated predictions
/// into `boost.<step>`. A final partial batch (the tail) is handled separately.
pub fn update_large<O, T, I>(
    data: &mut Dataset<T, I>,
    tree: Option<&Tree>,
    path: &Path,
    config: &UpdateConfig,
    rng: &mut StdRng,
    parallel: bool,
) -> Result<()>
where
    O: Objective,
    T: Copy + Into<usize> + IntoBytes + FromBytes + Send + Sync,
    I: RowIndexT,
{
    // The sampled rows per batch must scale with the dataset size, so the
    // in-memory `data(inputs)` view and the on-disk layout stay consistent.
    assert!(data.is_valid());
    assert!(config.bits_per_batch > 0 && config.bits_per_batch <= BATCH);
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
    let batch_count = rows.div_ceil(config.bits_per_batch);
    let last_batch = rows - rest;

    // Reshape the in-memory data from per-column full-row vectors into per-batch
    // views: `input_slices[b][c]` is batch `b`'s slice of feature column `c`.
    let mut input_slices: Vec<Vec<&mut [T]>> = Vec::with_capacity(batch_count);
    for _ in 0..batch_count {
        input_slices.push(vec![]);
    }
    for items in data.inputs.iter_mut() {
        for (src, dst) in items
            .chunks_mut(config.bits_per_batch)
            .zip(input_slices.iter_mut())
        {
            dst.push(src);
        }
    }

    // Mirror the same per-batch split for the row-index slice, so each batch
    // writes its derivatives into the matching `data.index` window.
    let index_slices = data
        .index
        .chunks_mut(config.bits_per_batch)
        .collect::<Vec<_>>();

    // Current boosting step is the one being produced: previous step + 1,
    // or 0 for the first step.
    let step = config.train_step.map_or(0, |n| n + 1);
    // Create and size the prediction file for this step up front so the
    // output is a fixed-length, fully written artifact for all rows.
    let output = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path.join(FILENAME_PREDICT).with_extension(step.to_string()))?;
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
                    let (files, error) =
                        match UpdateFiles::open(path, &data.features, config.train_step) {
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
                        process_batch::<O, _, _>(buffers, &mut inputs, index, tree, &mask, offset);
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
        let mut files = UpdateFiles::open(path, &data.features, config.train_step)?;
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
            // sampled rows' derivatives into `input`/`index`, then write the
            // updated predictions out at this batch's slot in `boost.<step>`.
            files.read(&mut buffers, read_size, config.initial)?;
            process_batch::<O, _, _>(&mut buffers, &mut inputs, index, tree, &mask, offset);
            files.output.write_all(buffers.predict.as_bytes())?;
        }
        Ok(())
    }
}

/// Add the candidate tree's contribution to every row's prediction, then for
/// the rows selected by `mask` compute their loss derivatives and record them
/// in `row_index` at the global position `offset + idx`.
fn process_batch<O: Objective, T: Copy + Into<usize>, I: RowIndexT>(
    buffers: &mut UpdateBuffers<T>,
    inputs: &mut [&mut [T]],
    row_index: &mut [RowIndex<I>],
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
        let (gradient, hessian) = O::grad_hess(buffers.target[bit], buffers.predict[bit]);
        row_index[idx] = RowIndex::new(offset + idx, gradient, hessian);
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
    fn open(path: &Path, features: &[Feature], train_step: Option<usize>) -> Result<Self> {
        let input = features
            .iter()
            .map(|feature| {
                let name = feature.name.replace(['.', '/', '\\'], "_");
                OpenOptions::new()
                    .read(true)
                    .open(path.join(FILENAME_INPUT).with_extension(&name))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let target = OpenOptions::new()
            .read(true)
            .open(path.join(FILENAME_TARGET))?;

        let predict = train_step
            .map(|step| {
                OpenOptions::new()
                    .read(true)
                    .open(path.join(FILENAME_PREDICT).with_extension(step.to_string()))
            })
            .transpose()?;

        let step = train_step.map_or(0, |n| n + 1);
        let output = OpenOptions::new()
            .write(true)
            .open(path.join(FILENAME_PREDICT).with_extension(step.to_string()))?;

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
/// re-sampled across calls). The `index` field is persistent scratch that
/// backs the partial Fisher–Yates shuffle used below.
struct BitmaskGenerator {
    index: Vec<u16>,
    total: usize,
}

impl BitmaskGenerator {
    /// Create a bitmask generator that samples within `total` rows
    /// (must be `<= BATCH`), initializing the persistent shuffle scratch.
    fn new(total: usize) -> Self {
        assert!(total <= BATCH);
        Self {
            index: (0..total).map(|i| i as u16).collect(),
            total,
        }
    }

    /// Return a bitmask of `bits` uniformly-randomly chosen rows out of
    /// `total`. Successive calls are independent: each one picks a fresh
    /// random subset of exactly `bits` rows (no replacement within a call),
    /// and `rng` only needs to provide `bits` (or `total - bits`) random
    /// integers. The retained `index` scratch is reused across calls, but
    /// Fisher–Yates yields a uniform subset from any starting permutation,
    /// so the residual state does not bias the result. When the chosen set is
    /// the majority, it is cheaper to invert (start all-ones and clear the
    /// unchosen).
    fn next(&mut self, rng: &mut StdRng, bits: usize) -> Vec<u64> {
        assert!(bits <= self.total);
        if bits <= self.total / 2 {
            // Partial Fisher–Yates: iteratively pick a random pivot in
            // `[i, total)`, swap it into slot `i`, and record the row id now
            // sitting at slot `i` as the chosen bit.
            //
            // Start from an empty mask and set the chosen bits.
            let mut mask = vec![0; self.total.div_ceil(64)];
            for i in 0..bits {
                let j = rng.random_range(i..self.total);
                self.index.swap(i, j);
                let (m, n) = (self.index[i] >> 6, self.index[i] & 63);
                mask[m as usize] |= 1 << n;
            }
            mask
        } else {
            // Same shuffle as above, but recording the *unchosen* rows so we
            // only need `total - bits` draws.
            //
            // Chosen set is a majority: start all-ones and clear the unchosen.
            let mut mask = vec![u64::MAX; self.total.div_ceil(64)];
            for i in 0..(self.total - bits) {
                let j = rng.random_range(i..self.total);
                self.index.swap(i, j);
                let (m, n) = (self.index[i] >> 6, self.index[i] & 63);
                mask[m as usize] &= !(1 << n);
            }
            // `u64::MAX` seed set every bit including the spurious ones past
            // `total`; mask the last word down to just the valid `last_bits`
            // so rows beyond the batch aren't accidentally marked.
            if !self.total.is_multiple_of(64) {
                let last_bits = self.total % 64;
                *mask.last_mut().unwrap() &= (1 << last_bits) - 1;
            }
            mask
        }
    }
}
