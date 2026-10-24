//! Per-feature histogram of the accumulated loss derivatives: one
//! (gradient, hessian) sum per feature bin, computed from the dataset's
//! `RowIndex` values and used to score split candidates and leaf outputs.
//!
//! Structures:
//! - `Histogram`: all features' bins in a single flat buffer of
//!   (grad, hess) pairs, plus the `build` entry point.

use crate::dataset::{Dataset, Feature, IndexT, Objective, RowIndex};
use rayon::prelude::*;
use std::ops::{AddAssign, SubAssign};

/// A histogram of the gradient/hessian accumulated per feature bin.
///
/// `data` holds one (grad, hess) pair per bin, laid out feature-major;
/// `splits` gives the per-feature offsets, so feature `c` is
/// `data[splits[c]..splits[c + 1]]`.
#[derive(Clone)]
pub struct Histogram {
    /// One (grad, hess) pair per bin.
    pub data: Vec<(f64, f64)>,
    /// Per-feature start offsets into `data`.
    pub splits: Vec<usize>,
}

impl Histogram {
    /// Build an all-zero histogram for `features`, each feature covering
    /// `feat.bins.len() + 1` bins (the value bins plus the missing bin).
    pub fn new(features: &[Feature]) -> Self {
        let splits = std::iter::once(0)
            .chain(features.iter().map(|feat| feat.bins.len() + 1))
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

    /// The (grad, hess) pairs of feature `column`.
    pub fn get(&self, column: usize) -> &[(f64, f64)] {
        let range = self.splits[column]..self.splits[column + 1];
        &self.data[range]
    }

    /// Build the full histogram from `data`'s per-row `index` values.
    ///
    /// For every row and every feature, the row's bin is taken from the
    /// row's value in that feature (`row[index][col]`), and the row's
    /// gradient/hessian (from `index`) is added into that bin. Each
    /// feature's rows are processed independently, so the result is
    /// identical for sequential and parallel execution. If `initial`
    /// is set, the `index` must be sequentially increasing.
    ///
    /// `parallel` partitions the rows into chunks and folds each chunk
    /// into its own histogram, then sums the partials into the result.
    pub fn build<T: Copy + Into<usize> + Sync, I: IndexT, O: Objective>(
        data: &Dataset<T>,
        index: &[RowIndex<I, O>],
        initial: bool,
        parallel: bool,
    ) -> Self {
        assert!(data.columns() > 0 && data.rows() >= index.len());

        let mut result = Self::new(&data.features);
        if parallel {
            const TASK_FACTOR: usize = 1024;
            let total_bins = result.data.len();
            let batch_size = total_bins.div_ceil(data.columns()) * TASK_FACTOR;
            let batch_count = data.rows().div_ceil(batch_size);

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
                            .zip(data.inputs.iter())
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
                    .zip(data.inputs.par_iter())
                    .for_each(|(slice, row)| {
                        build_line(slice, row, index, initial);
                    });
            }
        } else {
            // Sequential: features are built one by one, each row landing
            // into its bin for the current feature.
            for ([start, end], row) in result.splits.array_windows().zip(&data.inputs) {
                build_line(&mut result.data[*start..*end], row, index, initial);
            }
        }
        result
    }
}

impl AddAssign<&[(f64, f64)]> for Histogram {
    /// Add a full histogram into `self` bin by bin.
    fn add_assign(&mut self, rhs: &[(f64, f64)]) {
        for (dst, src) in self.data.iter_mut().zip(rhs) {
            *dst = (dst.0 + src.0, dst.1 + src.1);
        }
    }
}

impl SubAssign<&[(f64, f64)]> for Histogram {
    /// Subtract a full histogram from `self` bin by bin.
    fn sub_assign(&mut self, rhs: &[(f64, f64)]) {
        for (dst, src) in self.data.iter_mut().zip(rhs) {
            *dst = (dst.0 - src.0, dst.1 - src.1);
        }
    }
}

/// Accumulate the gradient/hessian of every row in `index` (a row chunk)
/// into `result`, one feature's slice. Each row lands in the bin of its
/// value for this feature: `result[row[item.index]]`.
fn build_line<T: Copy + Into<usize>, I: IndexT, O: Objective>(
    result: &mut [(f64, f64)],
    row: &[T],
    index: &[RowIndex<I, O>],
    initial: bool,
) {
    let mut update = |idx: T, (grad, hess): (f32, f32)| {
        let bin = &mut result[idx.into()];
        bin.0 += f64::from(grad);
        bin.1 += f64::from(hess);
    };
    if initial {
        let offset = index[0].index.to_usize();
        for (item, &hpos) in index.iter().zip(row.iter().skip(offset)) {
            update(hpos, item.deriv.grad_hess());
        }
    } else {
        for item in index {
            let hpos = row[item.index.to_usize()];
            update(hpos, item.deriv.grad_hess());
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
    use crate::dataset::SquaredLoss;
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

    // Build test dataset with the given size and the index.
    fn build_data_and_index(
        columns: usize,
        rows: usize,
    ) -> (Dataset<u8>, Vec<RowIndex<u32, SquaredLoss>>) {
        let data = Dataset::new(
            vec![Feature::linear("".into(), vec![0.0; 2]); columns],
            vec![[0, 1].repeat(rows / 2); columns],
            vec![0.0; rows],
            0.0,
        );
        let index = (0..rows)
            .map(|i| RowIndex::new(i, SquaredLoss::derive(1.0, 0.9)))
            .collect();
        (data, index)
    }

    // Floating-point compare helper.
    fn assert_almost_equal(actual: &[(f64, f64)], expected: &[(f64, f64)]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a.0 - b.0).abs() < 0.01);
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn histogram_simple() {
        let (data, index) = build_data_and_index(/*columns*/ 1, /*rows*/ 64);
        let hist = Histogram::build(&data, &index, true, /*parallel*/ false);
        assert_almost_equal(hist.get(0), &[(-3.2, 32.), (-3.2, 32.), (0., 0.)]);
    }

    #[test]
    fn histogram_parallel() {
        let (data, index) = build_data_and_index(/*columns*/ 2, /*rows*/ 20);
        let hist = Histogram::build(&data, &index, true, /*parallel*/ true);
        assert_almost_equal(hist.get(0), &[(-1., 10.), (-1., 10.), (0., 0.)]);
        assert_eq!(hist.get(1), hist.get(0));
    }

    #[test]
    fn parallel_split() {
        let (data, index) = build_data_and_index(/*columns*/ 3, /*rows*/ 100_000);
        let hist = Histogram::build(&data, &index, true, /*parallel*/ true);
        assert_almost_equal(hist.get(2), &[(-5e3, 5e4), (-5e3, 5e4), (0., 0.)]);
    }

    #[test]
    fn unordered_index() {
        let (data, mut index) = build_data_and_index(/*columns*/ 1, /*rows*/ 64);
        let mut rng = StdRng::seed_from_u64(0);
        index.shuffle(&mut rng);
        let hist = Histogram::build(&data, &index, /*initial*/ false, false);
        assert_almost_equal(hist.get(0), &[(-3.2, 32.), (-3.2, 32.), (0., 0.)]);
    }
}
