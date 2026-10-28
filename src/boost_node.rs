//! Split search and application for one boosting tree node: score candidate
//! splits against the node's histogram and partition the node's rows.
//!
//! Structures:
//! - `BoostNodeOptions`: regularization and split thresholds.
//! - `Splittable`: a node's rows, histogram, and stats - the thing that splits.
//! - `SplitCandidate` / `SplitStats`: a chosen split and one side's gain.
//!
//! Functions:
//! - `find_split`: find best split in one feature's histogram.
//! - `find_all_splits`: best split for each node across its candidate columns.
//! - `calc_gain`: a split side's regularized gain, `apply_l1(grad)^2 / (hess + l2)`.
//! - `apply_l1`: soft-threshold the gradient by the L1 regularizer.

use crate::dataset::{Dataset, Feature, FeatureKind, IndexT, Objective, RowIndex};
use crate::histogram::Histogram;
use itertools::{Itertools, partition};
use rayon::prelude::*;
use std::collections::HashSet;

/// Options controlling how a node is split.
#[derive(Clone, Debug)]
pub struct BoostNodeOptions {
    /// L1 regularization (alpha).
    pub reg_alpha: f64,
    /// L2 regularization (lambda).
    pub reg_lambda: f64,
    /// Minimum gain to perform a split.
    pub min_split_gain: f64,
    /// Minimum hessian sum in one leaf.
    pub min_child_weight: f64,
    /// Allow missing values to create a separate split.
    pub allow_missing_splits: bool,
}

impl Default for BoostNodeOptions {
    fn default() -> Self {
        Self {
            reg_alpha: 0.0,
            reg_lambda: 0.0,
            min_split_gain: 0.0,
            min_child_weight: 1.0,
            allow_missing_splits: false,
        }
    }
}

/// One node's state: the rows that reach it, the histogram accumulated over
/// those rows (per bin of each feature), and the node's own statistics.
pub(crate) struct Splittable<'a, I: IndexT, O: Objective> {
    /// The rows routed to this node, in arbitrary order.
    pub index: &'a mut [RowIndex<I, O>],
    /// Per-(feature, bin) gradient/hessian sums over `index`.
    pub histogram: Histogram,
    /// The node's aggregate statistics (including its own gain term).
    pub stats: SplitStats,
}

/// A candidate split of one feature: its bin, whether it folds missing rows
/// into the positive side, the resulting side statistics, and the gain.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SplitCandidate {
    /// Feature column this split tests.
    pub feature_id: usize,
    /// Target bin the row must match, 1-indexed (0 means undefined).
    pub bin_id: usize,
    /// `true` when missing rows route to the positive side.
    pub with_missing: bool,
    /// Positive-side statistics.
    pub positive: SplitStats,
    /// Negative-side statistics.
    pub negative: SplitStats,
    /// Improvement in the objective gained by this split.
    pub gain: f64,
}

/// One split side's aggregate: the gradient/hessian sums and the
/// regularized gain term they drive.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SplitStats {
    /// Sum of the rows in this side's gradient.
    pub gradient: f64,
    /// Sum of the rows in this side's hessian.
    pub hessian: f64,
    /// `apply_l1(grad)^2 / (hess + l2)` - the side's contribution to gain.
    pub gain: f64,
}

/// Find the best split in one feature's histogram.
///
/// Scans each bin in turn, accumulating the running left/right gradient and
/// hessian sums (a cumulative prefix for linear features, a single bin for
/// one-hot), and evaluates a split when both sides carry enough weight. If
/// moving missing rows to the positive side improves the combined gain (or
/// the positive side is too light without them), that variant wins.
///
/// `feature_id` is filled in by the caller per feature, and `bin_id` is
/// 1-indexed (compare against `Feature::bins[bin_id - 1]`).
pub(crate) fn find_split(
    histogram: &[(f64, f64)],
    stats: &SplitStats,
    options: &BoostNodeOptions,
    feature_kind: FeatureKind,
) -> Option<SplitCandidate> {
    let mut result: Option<SplitCandidate> = None;
    let make_stats = |(grad, hess)| {
        let gain = calc_gain(grad, hess, options.reg_alpha, options.reg_lambda);
        SplitStats::new(grad, hess, gain)
    };

    let epsilon = f32::MIN_POSITIVE as f64;
    let min_size = options.min_child_weight.max(epsilon);
    let min_gain = (options.min_split_gain - epsilon).max(0.0);

    let undef = histogram[0];
    let total = (stats.gradient, stats.hessian);
    if total.1 < min_size * 2.0 {
        return None; // no further splits are possible
    }

    // Skip bin 0 (missing) unless missing splits are allowed, and stop before
    // the last bin for linear features so the top bin cannot be split off.
    let start = if options.allow_missing_splits { 0 } else { 1 };
    let (end, is_linear) = match feature_kind {
        FeatureKind::OneHot => (histogram.len(), false),
        FeatureKind::Linear => (histogram.len() - 1, true),
    };

    // Running left/right sums as the split point walks the bins; `left` is
    // rebuilt from scratch for each one-hot bin and grown cumulatively for
    // each linear bin past the first. A bin with zero hessian is skipped -
    // with the objectives in `dataset`, zero hessian implies zero gradient,
    // so it contributes nothing to any split.
    let (mut left, mut right) = ((0.0, 0.0), total);
    for (bin_id, (grad, hess)) in histogram.iter().enumerate().take(end).skip(start) {
        if *hess == 0.0 {
            continue;
        }
        (left, right) = if is_linear && bin_id > 1 {
            (
                (left.0 + grad, left.1 + hess),
                (right.0 - grad, right.1 - hess),
            )
        } else {
            ((*grad, *hess), (total.0 - grad, total.1 - hess))
        };
        // The side the positive rows land on must carry enough weight; the
        // missing rows that join it (see below) are included in `left` when
        // evaluating that check.
        if left.1 + undef.1 < min_size || right.1 < min_size {
            continue;
        }

        let mut positive = make_stats(left);
        let mut negative = make_stats(right);
        let mut with_missing = bin_id == 0;

        // Try routing the missing rows into the positive side. The split is
        // allowed only when the negative side stays heavy enough, and it
        // takes over either unconditionally when it fixes an under-weight
        // positive side, or when it raises the combined gain.
        let left_too_small = left.1 < min_size;
        if undef.1 > 0.0 && right.1 - undef.1 >= min_size && !with_missing {
            let positive_alt = make_stats((left.0 + undef.0, left.1 + undef.1));
            let negative_alt = make_stats((right.0 - undef.0, right.1 - undef.1));
            if left_too_small
                || positive_alt.gain + negative_alt.gain > positive.gain + negative.gain
            {
                positive = positive_alt;
                negative = negative_alt;
                with_missing = true;
            }
        } else if left_too_small {
            continue;
        }

        // A split must raise the objective: its combined side gain, minus
        // this node's own gain term, must beat the best candidate so far
        // and the minimum split gain, and be strictly positive.
        let gain = positive.gain + negative.gain - stats.gain;
        let best = result.as_ref().map_or(min_gain, |x| x.gain);
        if gain > best {
            result = Some(SplitCandidate {
                // Set by the caller to the feature being scanned.
                feature_id: 0,
                bin_id,
                with_missing,
                positive,
                negative,
                gain,
            })
        }
    }
    result
}

/// Find the best split for each node.
///
/// Every `Splittable` is scanned across its `columns`, scoring each feature's
/// candidate via `find_split`; the highest-gain candidate becomes that node's
/// result. The return value maps 1:1 to `splittables` (same order), with
/// `None` for a node that yields no candidate clearing `find_split`'s
/// `min_child_weight` / `min_split_gain` gates.
///
/// `parallel` only changes how that work is distributed across threads, never
/// which candidate is chosen.
pub(crate) fn find_all_splits<I: IndexT, O: Objective>(
    splittables: Vec<&Splittable<I, O>>,
    features: &[Feature],
    options: &BoostNodeOptions,
    parallel: bool,
) -> Vec<Option<SplitCandidate>> {
    if parallel {
        // One task per (splittable, column) pair; `idx` is monotonically
        // non-decreasing across the flattened list, so the gather below may
        // chunk by it.
        let tasks = splittables
            .iter()
            .enumerate()
            .flat_map(|(idx, item)| item.histogram.iter().map(move |col| (idx, col)))
            .collect::<Vec<_>>();

        // Score each pair independently (no shared state), keeping only
        // splittables that found at least one candidate.
        let candidates = tasks
            .into_par_iter()
            .filter_map(|(idx, col)| {
                let item = splittables[idx];
                let (hist, feat) = (item.histogram.get(col), &features[col]);
                let mut res = find_split(hist, &item.stats, options, feat.kind())?;
                res.feature_id = col;
                Some((idx, res))
            })
            .collect::<Vec<_>>();

        // Fold the partials back per splittable: best candidate per splittable
        // wins, splittables with no candidate stay `None` (the 1:1 mapping to
        // the input order is preserved).
        let mut results = vec![None; splittables.len()];
        for (idx, group) in &candidates.into_iter().chunk_by(|(idx, _)| *idx) {
            results[idx] = group
                .map(|(_, res)| res)
                .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap());
        }
        results
    } else {
        // Sequential equivalent: same per-splittable best-gain candidate,
        // kept in input order (a splittable with no candidate maps to `None`).
        let find_best_split = |item: &Splittable<I, O>| {
            item.histogram
                .iter()
                .filter_map(|col| {
                    let (hist, feat) = (item.histogram.get(col), &features[col]);
                    let mut res = find_split(hist, &item.stats, options, feat.kind())?;
                    res.feature_id = col;
                    Some(res)
                })
                .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap())
        };
        splittables.into_iter().map(find_best_split).collect()
    }
}

impl<'a, I: IndexT, O: Objective> Splittable<'a, I, O> {
    /// Apply `split` to this node: partition the rows down the chosen bin,
    /// then derive each child's histogram (either by building the smaller
    /// side from scratch and subtracting it from the parent, or by
    /// discarding it with an empty histogram on the final split).
    ///
    /// Returns `(left, right)` where `left` is the positive side.
    pub fn split<T: Copy + Into<usize> + Sync>(
        mut self,
        split: SplitCandidate,
        data: &Dataset<T>,
        final_split: bool,
        parallel: bool,
    ) -> (Self, Self) {
        let feature = &data.features()[split.feature_id];
        let row = &data.inputs()[split.feature_id];

        // Move the positive rows to the front of `index`, returning the cut
        // position that separates the two sides.
        let pos = match feature.kind() {
            FeatureKind::OneHot => {
                partition_onehot(self.index, row, split.bin_id, split.with_missing)
            }
            FeatureKind::Linear => {
                partition_linear(self.index, row, split.bin_id, split.with_missing)
            }
        };
        let (index_left, index_right) = self.index.split_at_mut(pos);

        // On a non-final split, keep the larger side's existing histogram and
        // rebuild the smaller side from scratch, subtracting it to recover
        // the other side's histogram - cheaper than building both.
        let columns = || self.histogram.iter().collect::<HashSet<_>>();

        let (hist_left, hist_right) = if final_split {
            (Histogram::new(&[], None), Histogram::new(&[], None))
        } else if index_left.len() < index_right.len() {
            let hist = Histogram::build(data, Some(&columns()), index_left, false, parallel);
            self.histogram -= hist.data();
            (hist, self.histogram)
        } else {
            let hist = Histogram::build(data, Some(&columns()), index_right, false, parallel);
            self.histogram -= hist.data();
            (self.histogram, hist)
        };

        let left = Self {
            index: index_left,
            histogram: hist_left,
            stats: split.positive,
        };
        let right = Self {
            index: index_right,
            histogram: hist_right,
            stats: split.negative,
        };
        (left, right)
    }
}

impl SplitStats {
    pub fn new(gradient: f64, hessian: f64, gain: f64) -> Self {
        Self {
            gradient,
            hessian,
            gain,
        }
    }
}

/// Calculate regularized gain.
pub fn calc_gain(gradient: f64, hessian: f64, l1: f64, l2: f64) -> f64 {
    apply_l1(gradient, l1).powi(2) / (hessian + l2)
}

/// Soft-threshold `gradient` by `l1` - the L1-regularized gradient.
pub fn apply_l1(gradient: f64, l1: f64) -> f64 {
    if l1 == 0.0 {
        gradient
    } else if gradient.abs() > l1 {
        gradient - l1.copysign(gradient)
    } else {
        0.0
    }
}

/// Partition `index` in place so its positive rows (the bin that matches the
/// one-hot `value`, plus the missing bin when `with_missing`) lead, returning
/// the cut position.
fn partition_onehot<T: Copy + Into<usize>, I: IndexT, O: Objective>(
    index: &mut [RowIndex<I, O>],
    row: &[T],
    value: usize,
    with_missing: bool,
) -> usize {
    if with_missing {
        partition(index, |item| {
            let bin = row[item.as_usize()].into();
            bin == value || bin == 0
        })
    } else {
        partition(index, |item| {
            let bin = row[item.as_usize()].into();
            bin == value
        })
    }
}

/// Partition `index` in place so its positive rows (bins `1..=limit`, plus
/// the missing bin when `with_missing`) lead, returning the cut position.
fn partition_linear<T: Copy + Into<usize>, I: IndexT, O: Objective>(
    index: &mut [RowIndex<I, O>],
    row: &[T],
    limit: usize,
    with_missing: bool,
) -> usize {
    if with_missing {
        // Missing (bin 0) is `<= limit`, so it lands with the positive side.
        partition(index, |item| {
            let bin = row[item.as_usize()].into();
            bin <= limit
        })
    } else {
        // The missing bin wraps to `usize::MAX`, so the value bins are
        // exactly `1..=limit` and missing rows stay on the negative side.
        partition(index, |item| {
            let bin = row[item.as_usize()].into();
            bin.overflowing_sub(1).0 < limit
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::SquaredLoss;

    // Build test histogram: 3 columns, 5 rows, 3+1 bins.
    fn build_splittable() -> Splittable<'static, u32, SquaredLoss> {
        let feature = Feature::linear("".into(), vec![1., 2., 3.]);
        let mut histogram = Histogram::new(&vec![feature; 3], None);

        let (v0, v1, v2, v3) = ((0., 0.), (0.25, 1.), (0.5, 2.), (0.75, 3.));
        histogram
            .data_mut()
            .copy_from_slice(&[v0, v1, v2, v2, v1, v1, v2, v1, v3, v1, v1, v0]);
        let stats = SplitStats {
            gradient: 1.25,
            hessian: 5.0,
            gain: 0.0,
        };
        Splittable {
            index: &mut [],
            histogram,
            stats,
        }
    }

    // Find best split helper.
    fn test_find_split(
        splittable: Splittable<'_, u32, SquaredLoss>,
        features: &[Feature],
        options: &BoostNodeOptions,
    ) -> Option<SplitCandidate> {
        let mut result = find_all_splits(vec![&splittable], features, options, false);
        result.pop().unwrap()
    }

    #[test]
    fn linear_simple() {
        let features = vec![Feature::default(); 3];
        let options = BoostNodeOptions {
            reg_alpha: 0.25,
            reg_lambda: 0.1,
            ..Default::default()
        };
        let expected_gain = 0.13719512195121952;
        assert_eq!(
            test_find_split(build_splittable(), &features, &options),
            Some(SplitCandidate {
                feature_id: 2,
                bin_id: 1,
                with_missing: false,
                positive: SplitStats::new(0.25, 1.0, 0.0),
                negative: SplitStats::new(1.0, 4.0, expected_gain),
                gain: expected_gain,
            })
        );
    }

    #[test]
    fn linear_min_split_gain() {
        let features = vec![Feature::default(); 3];
        let options = |v: f64| BoostNodeOptions {
            min_split_gain: v,
            ..Default::default()
        };
        let res1 = test_find_split(build_splittable(), &features, &options(0.32));
        assert!(res1.is_none(), "{res1:?}");

        let res2 = test_find_split(build_splittable(), &features, &options(0.3));
        assert!(res2.is_some(), "{res2:?}");
        assert_eq!(res2.unwrap().gain, 0.3125);
    }

    #[test]
    fn linear_min_child_weight() {
        let features = vec![Feature::default(); 3];
        let options = |v: f64| BoostNodeOptions {
            min_child_weight: v,
            ..Default::default()
        };
        let mut splittable = build_splittable();
        splittable.histogram.data_mut()[2].0 = -1.0;
        assert_eq!(
            test_find_split(splittable, &features, &options(2.0)),
            Some(SplitCandidate {
                feature_id: 0,
                bin_id: 2,
                with_missing: false,
                positive: SplitStats::new(-0.75, 3.0, 0.1875),
                negative: SplitStats::new(2.0, 2.0, 2.0),
                gain: 2.1875,
            })
        );

        let too_large = test_find_split(build_splittable(), &features, &options(5.0));
        assert!(too_large.is_none(), "{too_large:?}");
    }

    #[test]
    fn linear_allow_missing_splits() {
        let features = vec![Feature::default(); 3];
        let options = BoostNodeOptions {
            allow_missing_splits: true,
            ..Default::default()
        };
        let mut splittable = build_splittable();
        splittable.histogram.data_mut()[4].0 = -1.0;
        assert_eq!(
            test_find_split(splittable, &features, &options),
            Some(SplitCandidate {
                feature_id: 1,
                bin_id: 0,
                with_missing: true,
                positive: SplitStats::new(-1.0, 1.0, 1.0),
                negative: SplitStats::new(2.25, 4.0, 1.265625),
                gain: 2.265625,
            })
        );
    }

    #[test]
    fn onehot_simple() {
        let features = vec![Feature::onehot("".into(), vec![]); 3];
        let mut splittable = build_splittable();
        splittable.histogram.data_mut()[4].0 = -2.0;
        splittable.histogram.data_mut()[7].0 = -2.0;
        assert_eq!(
            test_find_split(splittable, &features, &BoostNodeOptions::default()),
            Some(SplitCandidate {
                feature_id: 1,
                bin_id: 3,
                with_missing: true,
                positive: SplitStats::new(-4.0, 2.0, 8.0),
                negative: SplitStats::new(5.25, 3.0, 9.1875),
                gain: 17.1875,
            })
        );
    }

    #[test]
    fn search_partial() {
        let features = vec![Feature::default(); 3];
        let mut splittable = build_splittable();
        splittable.histogram.compress(&HashSet::from([0, 1]));
        assert_eq!(
            test_find_split(splittable, &features, &BoostNodeOptions::default()),
            Some(SplitCandidate {
                feature_id: 1,
                bin_id: 1,
                with_missing: false,
                positive: SplitStats::new(0.25, 1.0, 0.0625),
                negative: SplitStats::new(1.0, 4.0, 0.25),
                gain: 0.3125,
            })
        );
    }

    #[test]
    fn search_parallel() {
        let features = vec![Feature::default(); 3];
        let result = find_all_splits(
            vec![&build_splittable()],
            &features,
            &BoostNodeOptions::default(),
            /*parallel*/ true,
        );
        assert_eq!(
            *result.last().unwrap(),
            Some(SplitCandidate {
                feature_id: 2,
                bin_id: 1,
                with_missing: false,
                positive: SplitStats::new(0.25, 1.0, 0.0625),
                negative: SplitStats::new(1.0, 4.0, 0.25),
                gain: 0.3125,
            })
        );
    }

    #[test]
    fn regularization() {
        assert_eq!(apply_l1(-0.5, 0.3), -0.2);
        assert_eq!(apply_l1(-0.25, 0.3), 0.0);
        assert_eq!(apply_l1(0.25, 0.3), 0.0);
        assert_eq!(apply_l1(0.5, 0.3), 0.2);
        assert_eq!(calc_gain(2.0, 1.0, 0.2, 0.2), 2.7);
    }

    #[test]
    fn partition_onehot() {
        let row = vec![0_u8, 1, 2, 3, 4];
        let mut index = RowIndex::<u32, SquaredLoss>::new_array(5);
        assert_eq!(super::partition_onehot(&mut index, &row, 3, false), 1);
        assert_eq!(index[0].as_usize(), 3);

        index.sort_by_key(|x| x.as_usize());
        assert_eq!(super::partition_onehot(&mut index, &row, 3, true), 2);
        assert!(index[..2].iter().all(|x| [0, 3].contains(&x.as_usize())));
    }

    #[test]
    fn partition_linear() {
        let row = vec![0_u8, 1, 2, 3, 4];
        let mut index = RowIndex::<u32, SquaredLoss>::new_array(5);
        assert_eq!(super::partition_linear(&mut index, &row, 2, false), 2);
        assert!(index[..2].iter().all(|x| [1, 2].contains(&x.as_usize())));

        index.sort_by_key(|x| x.as_usize());
        assert_eq!(super::partition_linear(&mut index, &row, 2, true), 3);
        assert!(index[..3].iter().all(|x| x.as_usize() <= 2));
    }

    #[test]
    fn split_node() {
        let data = Dataset::<u16>::new(
            vec![Feature::linear("".into(), vec![1., 2., 3.]); 3],
            vec![
                vec![3, 2, 1, 2, 3], // feature being split
                vec![2, 1, 0, 3, 2],
                vec![0, 1, 0, 2, 0],
            ],
            vec![],
        )
        .unwrap();

        let stats_1 = SplitStats::new(0.1, 0.2, 0.3);
        let stats_2 = SplitStats::new(0.4, 0.5, 0.6);
        let candidate = |bin_id: usize| SplitCandidate {
            feature_id: 0,
            bin_id,
            with_missing: false,
            positive: stats_1.clone(),
            negative: stats_2.clone(),
            gain: 0.0,
        };

        let make_index = |n: usize| -> Vec<RowIndex<u32, SquaredLoss>> {
            (0..n)
                .map(|i| RowIndex::new(i, SquaredLoss::derive(0.0, 0.25)))
                .collect()
        };
        let make_hist = |data: &[u32]| -> Vec<(f64, f64)> {
            data.iter().map(|&x| (x as f64 * 0.25, x as f64)).collect()
        };

        // Left side is smaller.
        let splittable = Splittable {
            index: &mut make_index(5),
            ..build_splittable()
        };
        let (left, right) = splittable.split(candidate(1), &data, false, false);
        assert_eq!(left.index.len(), 1);
        assert_eq!(left.histogram.get(0), make_hist(&[0, 1, 0, 0]));
        assert_eq!(right.index.len(), 4);
        assert_eq!(right.histogram.get(0), make_hist(&[0, 0, 2, 2]));

        // Right side is smaller.
        let splittable = Splittable {
            index: &mut make_index(5),
            ..build_splittable()
        };
        let (left, right) = splittable.split(candidate(2), &data, false, false);
        assert_eq!(left.index.len(), 3);
        assert_eq!(left.histogram.get(0), make_hist(&[0, 1, 2, 0]));
        assert_eq!(right.index.len(), 2);
        assert_eq!(right.histogram.get(0), make_hist(&[0, 0, 0, 2]));

        // Final split (no histograms).
        let splittable = Splittable {
            index: &mut make_index(5),
            ..build_splittable()
        };
        let (left, right) = splittable.split(candidate(3), &data, true, true);
        assert!(left.histogram.data().is_empty());
        assert_eq!(left.stats, stats_1);
        assert!(right.histogram.data().is_empty());
        assert_eq!(right.stats, stats_2);
    }
}
