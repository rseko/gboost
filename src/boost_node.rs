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

use crate::dataset::{Dataset, Feature, FeatureType, IndexT, Objective, RowIndex};
use crate::histogram::Histogram;
use itertools::{Itertools, partition};
use rayon::prelude::*;

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
pub struct Splittable<'a, I: IndexT, O: Objective> {
    /// The rows routed to this node, in arbitrary order.
    pub index: &'a mut [RowIndex<I, O>],
    /// Per-(feature, bin) gradient/hessian sums over `index`.
    pub histogram: Histogram,
    /// The node's aggregate statistics (including its own gain term).
    pub stats: SplitStats,
    /// Feature index for selecting candidates.
    pub columns: Vec<usize>,
}

/// A candidate split of one feature: its bin, whether it folds missing rows
/// into the positive side, the resulting side statistics, and the gain.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitCandidate {
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
pub struct SplitStats {
    /// Sum of the rows in this side's gradient.
    pub gradient: f64,
    /// Sum of the rows in this side's hessian.
    pub hessian: f64,
    /// `apply_l1(grad).^2 / (hess + l2)` - the side's contribution to gain.
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
pub fn find_split(
    histogram: &[(f64, f64)],
    stats: &SplitStats,
    options: &BoostNodeOptions,
    feature_type: FeatureType,
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
    let (end, is_linear) = match feature_type {
        FeatureType::OneHot => (histogram.len(), false),
        FeatureType::Linear => (histogram.len() - 1, true),
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
pub fn find_all_splits<I: IndexT, O: Objective>(
    splittables: &[Splittable<I, O>],
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
            .flat_map(|(idx, item)| item.columns.iter().map(move |&col| (idx, col)))
            .collect::<Vec<_>>();

        // Score each pair independently (no shared state), keeping only
        // splittables that found at least one candidate.
        let candidates = tasks
            .into_par_iter()
            .filter_map(|(idx, col)| {
                let item = &splittables[idx];
                let (hist, feat) = (item.histogram.get(col), &features[col]);
                let mut res = find_split(hist, &item.stats, options, feat.type_)?;
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
            item.columns
                .iter()
                .filter_map(|&col| {
                    let (hist, feat) = (item.histogram.get(col), &features[col]);
                    let mut res = find_split(hist, &item.stats, options, feat.type_)?;
                    res.feature_id = col;
                    Some(res)
                })
                .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap())
        };
        splittables.iter().map(find_best_split).collect()
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
        let feature = &data.features[split.feature_id];
        let row = &data.inputs[split.feature_id];

        // Move the positive rows to the front of `index`, returning the cut
        // position that separates the two sides.
        let pos = match feature.type_ {
            FeatureType::OneHot => {
                partition_onehot(self.index, row, split.bin_id, split.with_missing)
            }
            FeatureType::Linear => {
                partition_linear(self.index, row, split.bin_id, split.with_missing)
            }
        };
        let (index_left, index_right) = self.index.split_at_mut(pos);

        // On a non-final split, keep the larger side's existing histogram and
        // rebuild the smaller side from scratch, subtracting it to recover
        // the other side's histogram - cheaper than building both.
        let (hist_left, hist_right) = if final_split {
            (Histogram::new(&[], &[]), Histogram::new(&[], &[]))
        } else if index_left.len() < index_right.len() {
            let hist = Histogram::build(data, &self.columns, index_left, false, parallel);
            self.histogram -= &hist.data;
            (hist, self.histogram)
        } else {
            let hist = Histogram::build(data, &self.columns, index_right, false, parallel);
            self.histogram -= &hist.data;
            (self.histogram, hist)
        };

        let left = Self {
            index: index_left,
            histogram: hist_left,
            stats: split.positive,
            columns: self.columns.clone(),
        };
        let right = Self {
            index: index_right,
            histogram: hist_right,
            stats: split.negative,
            columns: self.columns,
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
            let bin = row[item.index.to_usize()].into();
            bin == value || bin == 0
        })
    } else {
        partition(index, |item| {
            let bin = row[item.index.to_usize()].into();
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
            let bin = row[item.index.to_usize()].into();
            bin <= limit
        })
    } else {
        // The missing bin wraps to `usize::MAX`, so the value bins are
        // exactly `1..=limit` and missing rows stay on the negative side.
        partition(index, |item| {
            let bin = row[item.index.to_usize()].into();
            bin.overflowing_sub(1).0 < limit
        })
    }
}
