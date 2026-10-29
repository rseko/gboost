//! Per-feature histogram of the accumulated loss derivatives: one
//! (gradient, hessian) sum per feature bin, computed from the dataset's
//! `RowIndex` values and used to score split candidates and leaf outputs.
//!
//! Structures:
//! - `Histogram`: all features' bins in a single flat buffer of
//!   (grad, hess) pairs, plus the `build` entry point.

use crate::dataset::{Dataset, Feature, IndexT, Objective, RowIndex};
use rayon::prelude::*;
use std::collections::HashSet;
use std::ops::{AddAssign, SubAssign};

type Item = (f64, f64);

/// A histogram of the gradient/hessian accumulated per feature bin.
///
/// `data` holds one (grad, hess) pair per bin, laid out feature-major;
/// `splits` gives the per-feature offsets, so feature `c` is
/// `data[splits[c]..splits[c + 1]]`.
#[derive(Clone)]
pub struct Histogram {
    /// One (grad, hess) pair per bin.
    data: Vec<Item>,
    /// Per-feature start offsets into `data`.
    splits: Vec<usize>,
}

impl Histogram {
    /// Build an all-zero histogram for `features`, each feature covering
    /// `feat.bins.len() + 1` bins (the value bins plus the missing bin).
    /// Filter out features that are not present in `columns`.
    pub fn new(features: &[Feature], columns: Option<&HashSet<usize>>) -> Self {
        let splits = std::iter::once(0)
            .chain(features.iter().enumerate().map(|(idx, feat)| {
                if columns.is_none_or(|c| c.contains(&idx)) {
                    feat.bins().len() + 1
                } else {
                    0
                }
            }))
            .scan(0, |acc, size| {
                *acc += size;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        Self {
            data: vec![(0.0, 0.0); *splits.last().unwrap()],
            splits,
        }
    }

    /// Get full histogram data.
    pub fn data(&self) -> &[Item] {
        &self.data
    }

    /// The (grad, hess) pairs of feature `column`.
    pub fn get(&self, column: usize) -> &[Item] {
        let range = self.splits[column]..self.splits[column + 1];
        &self.data[range]
    }

    /// Return an iterator over non-empty column indices.
    pub fn iter(&self) -> HistogramIter<'_> {
        HistogramIter {
            splits: &self.splits,
            index: 0,
        }
    }

    /// Build the full histogram from `data`'s per-row `index` values.
    ///
    /// For every row and every feature, the row's bin is taken from the
    /// row's value in that feature (`row[index][col]`), and the row's
    /// gradient/hessian (from `index`) is added into that bin. Each
    /// feature's rows are processed independently, so the result is
    /// identical for sequential and parallel execution. If `initial`
    /// is set, the `index` must be consecutive (`index[i] == index[0] + i`).
    ///
    /// `parallel` partitions the rows into chunks and folds each chunk
    /// into its own histogram, then sums the partials into the result.
    pub fn build<T: Copy + Into<usize> + Sync, I: IndexT, O: Objective>(
        data: &Dataset<T>,
        columns: Option<&HashSet<usize>>,
        index: &[RowIndex<I, O>],
        initial: bool,
        parallel: bool,
    ) -> Self {
        let mut result = Self::new(data.features(), columns);
        assert!(!result.data.is_empty());
        assert!(data.rows() >= index.len());

        if parallel {
            const TASK_FACTOR: usize = 1024;
            // Calculate split-K batch size: rows are partitioned into
            // `batch_count` chunks, each folding into its own partial
            // histogram of `total_bins` cells, then summed into the result.
            // `batch_size` is the row count per chunk, heuristic-scaled
            // from the average bin count per contributing feature.
            let total_bins = result.data.len();
            let total_cols = columns.map_or(data.columns(), |c| c.len());
            let batch_size = total_bins.div_ceil(total_cols) * TASK_FACTOR;
            let batch_count = index.len().div_ceil(batch_size);

            if batch_count > 1 {
                // Many row chunks: each fold runs in parallel into a private
                // histogram inside a preallocated buffer (one per chunk), then
                // the partials are summed into the shared result.
                let mut buffer = vec![(0.0, 0.0); total_bins * batch_count];
                let tasks = buffer
                    .chunks_mut(total_bins)
                    .zip(index.chunks(batch_size))
                    .flat_map(|(hist, src)| {
                        split_parts(hist, &result.splits)
                            .into_iter()
                            .zip(data.inputs().iter())
                            .map(move |item| (item, src))
                    })
                    .collect::<Vec<_>>();

                tasks.into_par_iter().for_each(|((slice, row), part)| {
                    build_line(slice, row, part, initial);
                });
                for hist in buffer.chunks(total_bins) {
                    result += hist;
                }
            } else {
                // Single row chunk: every feature is built in parallel
                // directly into the result, no partials to fold.
                split_parts(&mut result.data, &result.splits)
                    .into_par_iter()
                    .zip(data.inputs().par_iter())
                    .for_each(|(slice, row)| {
                        build_line(slice, row, index, initial);
                    });
            }
        } else {
            // Sequential: features are built one by one, each row landing
            // into its bin for the current feature.
            for ([start, end], row) in result.splits.array_windows().zip(data.inputs()) {
                build_line(&mut result.data[*start..*end], row, index, initial);
            }
        }
        result
    }

    /// Mutable histogram data (internal).
    #[allow(unused)]
    pub(crate) fn data_mut(&mut self) -> &mut [Item] {
        &mut self.data
    }
}

impl AddAssign<&[Item]> for Histogram {
    /// Add a full histogram into `self` bin by bin.
    fn add_assign(&mut self, rhs: &[Item]) {
        assert_eq!(rhs.len(), self.data.len());
        for (dst, src) in self.data.iter_mut().zip(rhs) {
            *dst = (dst.0 + src.0, dst.1 + src.1);
        }
    }
}

impl SubAssign<&[Item]> for Histogram {
    /// Subtract a full histogram from `self` bin by bin.
    fn sub_assign(&mut self, rhs: &[Item]) {
        assert_eq!(rhs.len(), self.data.len());
        for (dst, src) in self.data.iter_mut().zip(rhs) {
            *dst = (dst.0 - src.0, dst.1 - src.1);
        }
    }
}

/// Simple iterator over non-empty column indices.
pub struct HistogramIter<'a> {
    splits: &'a [usize],
    index: usize,
}

impl<'a> Iterator for HistogramIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.index + 1 < self.splits.len() {
            self.index += 1;
            let (start, end) = (self.splits[self.index - 1], self.splits[self.index]);
            if start != end {
                return Some(self.index - 1);
            }
        }
        None
    }
}

/// Accumulate the gradient/hessian of every row in `index` (a row chunk)
/// into `result`, one feature's slice. Each row lands in the bin of its
/// value for this feature: `result[row[item.index]]`.
fn build_line<T: Copy + Into<usize>, I: IndexT, O: Objective>(
    result: &mut [Item],
    row: &[T],
    index: &[RowIndex<I, O>],
    initial: bool,
) {
    if result.is_empty() {
        return;
    }
    let mut update = |idx: T, (grad, hess): (f32, f32)| {
        let bin = &mut result[idx.into()];
        bin.0 += f64::from(grad);
        bin.1 += f64::from(hess);
    };

    if initial {
        let offset = index[0].as_usize();
        for (item, &hpos) in index.iter().zip(row.iter().skip(offset)) {
            update(hpos, item.grad_hess());
        }
    } else {
        for item in index {
            let hpos = row[item.as_usize()];
            update(hpos, item.grad_hess());
        }
    }
}

/// Split `slice` into the per-feature sub-slices given by `splits`,
/// preserving order. Used to address one feature's bins while building
/// each feature in parallel.
fn split_parts<'a, T: Copy>(mut slice: &'a mut [T], splits: &[usize]) -> Vec<&'a mut [T]> {
    let mut result = Vec::with_capacity(splits.len() - 1);
    for [start, end] in splits.array_windows() {
        let (head, tail) = slice.split_at_mut(*end - *start);
        result.push(head);
        slice = tail;
    }
    assert!(slice.is_empty());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{SquaredLoss, create_test_dataset};
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

    // Build histogram helper.
    fn build_histogram(mut data: Dataset<u16>, parallel: bool) -> Histogram {
        for slice in data.slices_mut() {
            for (k, v) in slice.iter_mut().enumerate() {
                *v = k as u16 % 2;
            }
        }
        let index = (0..data.rows())
            .map(|i| RowIndex::<u32, _>::new(i, SquaredLoss::derive(1.0, 0.9)))
            .collect::<Vec<_>>();
        Histogram::build(&data, None, &index, true, parallel)
    }

    // Floating-point compare helper.
    fn assert_almost_equal(actual: &[Item], expected: &[Item]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a.0 - b.0).abs() < 0.01, "{} != {}", a.0, b.0);
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn create_histogram() {
        let hist = build_histogram(create_test_dataset(3, 64, 2), /*parallel*/ false);
        assert_almost_equal(hist.get(0), &[(-3.2, 32.), (-3.2, 32.), (0., 0.)]);

        let hist = build_histogram(create_test_dataset(3, 100, 2), /*parallel*/ true);
        assert_almost_equal(hist.get(1), &[(-5., 50.), (-5., 50.), (0., 0.)]);

        let hist = build_histogram(create_test_dataset(3, 100_000, 2), /*parallel*/ true);
        assert_almost_equal(hist.get(2), &[(-5e3, 5e4), (-5e3, 5e4), (0., 0.)]);
    }

    #[test]
    fn create_partial() {
        let mut data = create_test_dataset(/*columns*/ 4, /*rows*/ 20, /*bins*/ 2);
        for (k, v) in data.slices_mut()[0].iter_mut().enumerate() {
            *v = k as u16 % 3;
        }
        let mut index = (0..data.rows())
            .map(|i| RowIndex::<u32, _>::new(i, SquaredLoss::derive(0.5, 0.6)))
            .collect::<Vec<_>>();

        let mut rng = StdRng::seed_from_u64(0);
        index.shuffle(&mut rng);
        index.truncate(15);
        let columns = HashSet::from([0, 2]);

        let hist = Histogram::build(&data, Some(&columns), &index, /*initial*/ false, true);
        assert_almost_equal(hist.get(0), &[(0.6, 6.), (0.5, 5.), (0.4, 4.)]);

        let hist_columns = Vec::from_iter(hist.iter());
        assert_eq!(hist_columns, vec![0, 2]);
    }

    #[test]
    fn slice_ops() {
        let hist = build_histogram(create_test_dataset(3, 64, 2), false);
        let mut other = hist.clone();
        other += hist.data();
        for (a, b) in other.data.iter().zip(hist.data.iter()) {
            assert_eq!(*a, (b.0 * 2.0, b.1 * 2.0));
        }
        other -= hist.data();
        assert_eq!(other.data(), hist.data());
    }
}
