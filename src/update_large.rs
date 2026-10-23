//! Incremental (staged) boost update for large, on-disk datasets.
//!
//! Rather than holding the full dataset in memory, this module streams row
//! batches from per-column files and re-scores them against the previous
//! model's predictions to compute fresh loss derivatives. Each boosting step
//! is persisted to `boost.<step>` so later steps can be read back as the base
//! prediction.

use crate::dataset::{Dataset, Feature, Objective, RowIndex, RowIndexT};
use crate::tree::Tree;
use anyhow::Result;
use rand::{RngExt, rngs::StdRng};
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
    config: UpdateConfig,
    rng: &mut StdRng,
) -> Result<()>
where
    O: Objective,
    T: Copy + Default + Into<usize> + IntoBytes + FromBytes,
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
    output.set_len((data.rows() * std::mem::size_of::<f32>()) as u64)?;

    // Reuse a single set of buffers for every batch to avoid repeated
    // allocation: inputs, target, and prediction are read into it in-place.
    let mut mask_generator = BitmaskGenerator::new(BATCH);
    let mut files = UpdateFiles::open::<T>(path, &data.features, config.train_step, 0)?;

    let mut shared_buffer = vec![T::default(); BATCH * data.columns()];
    let mut buffers = UpdateBuffers::new(&mut shared_buffer);

    // Mutable views into the in-memory feature columns. During sampling, the
    // chosen rows' values are copied here (at their global positions), while
    // the loss derivatives (grad/hess) go into `data.index` below.
    let mut inputs = data
        .inputs
        .iter_mut()
        .map(Vec::as_mut_slice)
        .collect::<Vec<_>>();

    // Main loop: for each full batch, sample `bits_per_batch` rows, read the
    // batch, add the tree contribution, recompute derivatives into `data.index`,
    // and write the updated predictions.
    let batch_count = config.total_rows / BATCH;
    for batch_no in 0..batch_count {
        let mask = mask_generator.next(rng, config.bits_per_batch);
        files.read(&mut buffers, BATCH, config.initial)?;
        process_batch::<O, _, _>(
            &mut buffers,
            &mut inputs,
            &mut data.index,
            tree,
            &mask,
            batch_no * config.bits_per_batch,
        );
        files.output.write_all(buffers.predict.as_bytes())?;
    }

    // Final partial batch: fewer than BATCH rows, so reseed the generator at
    // the tail size and only sample `rest` rows.
    if tail != 0 {
        let mask = BitmaskGenerator::new(tail).next(rng, rest);
        files.read(&mut buffers, tail, config.initial)?;
        buffers.predict.truncate(tail);
        process_batch::<O, _, _>(
            &mut buffers,
            &mut inputs,
            &mut data.index,
            tree,
            &mask,
            batch_count * config.bits_per_batch,
        );
        files.output.write_all(buffers.predict.as_bytes())?;
    }
    Ok(())
}

/// Process a single loaded batch: add the candidate tree's contribution to
/// every row's prediction, then (for the rows selected by `mask`) copy their
/// feature values into `inputs`, compute the loss gradient/hessian, and record
/// them in `row_index` at the global position `offset + idx`.
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
            let get_bin = |col: usize| buffers.input[col][idx].into();
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
        let pos = offset + idx; // global row position
        // Copy the sampled row's feature values from the batch buffer into the
        // in-memory inputs column view.
        for (src, dst) in buffers.input.iter().zip(inputs.iter_mut()) {
            dst[pos] = src[bit];
        }
        // Recompute the loss derivatives for this row using its true target and
        // the (possibly tree-adjusted) prediction, then store them.
        let (gradient, hessian) = O::grad_hess(buffers.target[bit], buffers.predict[bit]);
        row_index[pos] = RowIndex::new(pos, gradient, hessian);
    });
}

/// Scratch workspace holding one batch of data. `input` is a set of column
/// slices (one per feature) partitioned from a single shared buffer; `target`
/// and `predict` are the row-aligned label and running-prediction columns.
struct UpdateBuffers<'a, T> {
    input: Vec<&'a mut [T]>,
    target: Vec<f32>,
    predict: Vec<f32>,
}

impl<'a, T> UpdateBuffers<'a, T> {
    /// Build a workspace from a shared `BATCH * columns` buffer. The buffer is
    /// split into `columns` slices of `BATCH` rows (the input), and fresh
    /// `BATCH`-sized target/prediction columns are allocated.
    fn new(buffer: &'a mut [T]) -> Self {
        Self {
            input: buffer.chunks_mut(BATCH).collect(),
            target: vec![0.0; BATCH],
            predict: vec![0.0; BATCH],
        }
    }
}

/// Open file handles for a single on-disk dataset, positioned at a given
/// offset. `predict` is the previous step's prediction (if any) and `output`
/// is the current step's prediction file being written.
struct UpdateFiles {
    input: Vec<File>,
    target: File,
    predict: Option<File>,
    output: File,
}

impl UpdateFiles {
    fn open<T: Sized>(
        path: &Path,
        features: &[Feature],
        train_step: Option<usize>,
        offset: usize,
    ) -> Result<Self> {
        // Open one feature file per column, sanitizing the name.
        let mut input = features
            .iter()
            .map(|feature| {
                let name = feature.name.replace(['.', '/', '\\'], "_");
                OpenOptions::new()
                    .read(true)
                    .open(path.join(FILENAME_INPUT).with_extension(&name))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Seek each feature file to the row `offset` (bytes = offset * size_of::<T>).
        let position = (offset * std::mem::size_of::<T>()) as u64;
        for file in input.iter_mut() {
            file.seek(SeekFrom::Start(position))?;
        }

        // Target is f32, so the seek offset uses size_of::<f32> rather than T.
        let mut target = OpenOptions::new()
            .read(true)
            .open(path.join(FILENAME_TARGET))?;
        let position = (offset * std::mem::size_of::<f32>()) as u64;
        target.seek(SeekFrom::Start(position))?;

        // Previous-step prediction only exists if `train_step` is set;
        // otherwise `predict` stays `None` and `read` seeds via `initial`.
        let mut predict = train_step
            .map(|step| {
                OpenOptions::new()
                    .read(true)
                    .open(path.join(FILENAME_PREDICT).with_extension(step.to_string()))
            })
            .transpose()?;
        if let Some(file) = &mut predict {
            file.seek(SeekFrom::Start(position))?;
        }

        // Output is next step's prediction file (train_step + 1, or 0 on first step),
        // truncated and written by the streaming loop.
        let step = train_step.map_or(0, |n| n + 1);
        let mut output = OpenOptions::new()
            .write(true)
            .open(path.join(FILENAME_PREDICT).with_extension(step.to_string()))?;
        output.seek(SeekFrom::Start(position))?;

        Ok(Self {
            input,
            target,
            predict,
            output,
        })
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
        for (file, slice) in self.input.iter_mut().zip(buffers.input.iter_mut()) {
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
            // Chosen set is a majority: start all-ones and clear the unchosen.
            let mut mask = vec![u64::MAX; self.total.div_ceil(64)];
            for i in 0..(self.total - bits) {
                let j = rng.random_range(i..self.total);
                self.index.swap(i, j);
                let (m, n) = (self.index[i] >> 6, self.index[i] & 63);
                mask[m as usize] &= !(1 << n);
            }
            if !self.total.is_multiple_of(64) {
                let last_bits = self.total % 64;
                *mask.last_mut().unwrap() = (1 << last_bits) - 1;
            }
            mask
        }
    }
}
