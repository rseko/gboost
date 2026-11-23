//! Bin-boundary fitting: turn a feature's raw values into the sorted
//! thresholds the tree compares against, and map each value to its
//! 1-based bin index among those thresholds.
//!
//! - `quantize` maps a single value to its bin index; NaN maps to 0.
//! - `quantize_all` maps every column's values to their bin indices using
//!   pre-fitted boundaries.
//! - `bins_from` is the fitting entry point; NaNs are always dropped,
//!   and an empty result is normalized to a single positive infinity.
//!   - `fit_small` uses an exact float16 histogram (total ordering) and
//!     merges the distinct values into `max_bins` contiguous groups.
//!   - `fit_large` sorts the values and either keeps the distinct values
//!     or samples them evenly by rank.
//! - `bin_all` fits every column's bin boundaries.
//! - `QuantFloat` is the bound every public float in this module shares.

use log::trace;
use num_traits::{Float, PrimInt};
use rayon::prelude::*;
use std::collections::{BinaryHeap, VecDeque};
use zerocopy::{FromBytes, Immutable, IntoBytes};

/// Map a value to its 1-based bin index among the sorted `bins`.
/// With `bins = [1, 2, 3]`: `value <= 1` → 1, `1 < value <= 2` → 2,
/// `value > 2` → 3. NaN maps to 0.
/// Bins that are outside the range of `I` type also map to 0.
pub fn quantize<F: Float, I: PrimInt>(value: &F, bins: &[F]) -> I {
    if !value.is_nan() {
        let pos = bins.partition_point(|x| value > x) + 1;
        I::from(pos.min(bins.len())).unwrap_or(I::zero())
    } else {
        I::zero()
    }
}

/// Map each value of every column to its 1-based bin index among that
/// column's pre-fitted boundaries. `data_with_bins` pairs each column with
/// its `bins` (both in column order); `quantized` is one `Vec<I>` per
/// column (1-based, NaN -> 0).
pub fn quantize_all<F: Float + Sync, I: PrimInt + Send>(
    data_with_bins: &[(&[F], &[F])],
    parallel: bool,
) -> Vec<Vec<I>> {
    trace!("quantize_all: columns={}", data_with_bins.len());

    let quantize = |(data, bins): &(&[F], &[F])| -> Vec<I> {
        data.iter().map(|x| quantize(x, bins)).collect()
    };
    if parallel {
        data_with_bins.par_iter().map(quantize).collect()
    } else {
        data_with_bins.iter().map(quantize).collect()
    }
}

/// A floating-point type suitable for quantization.
pub trait QuantFloat: Float + FromBytes + Immutable + IntoBytes + Send + Sync {}
impl<T: Float + FromBytes + Immutable + IntoBytes + Send + Sync> QuantFloat for T {}

/// Fit a quantizer per column. Returns the fitted boundaries, one `Vec<F>`
/// per column in column order.
pub fn bin_all<F: QuantFloat>(
    data: &[&[F]],
    max_bins: usize,
    oversample: usize,
    parallel: bool,
) -> Vec<Vec<F>> {
    trace!("bin_all: columns={} max_bins={}", data.len(), max_bins);

    let bin = |data: &&[F]| bins_from(data, max_bins, oversample);
    if parallel {
        data.par_iter().map(bin).collect()
    } else {
        data.iter().map(bin).collect()
    }
}

/// Fit up to `max_bins` boundaries from `data`.
///
/// NaNs are never treated as values. Dispatches to the exact histogram
/// algorithm for float16, otherwise to the sort-based one. `oversample`
/// only affects the sort-based path. The returned `bins` is always
/// non-empty.
pub fn bins_from<F: QuantFloat>(data: &[F], max_bins: usize, oversample: usize) -> Vec<F> {
    assert!(max_bins > 0);

    let n_bytes = std::mem::size_of::<F>();
    let mut bins = if n_bytes == 2 && data.len() > max_bins {
        trace!("using histogram binning");
        fit_small::<_, u16>(data, max_bins)
    } else {
        trace!("using sort binning (sample={})", oversample);
        fit_large(data, max_bins, oversample.max(1))
    };

    // A column with no usable values still needs one bin to split on.
    if bins.is_empty() {
        bins.push(F::infinity());
    }
    bins
}

/// Histogram-based fit for a float type `T` whose values fit in `I`.
///
/// Exact: each distinct value occupies one histogram slot, ordered by
/// `to_bits`. If the distinct count is within `max_bins` every distinct
/// value is kept; otherwise the non-zero slot counts are merged into
/// `max_bins` contiguous groups and the last value of each group is
/// emitted as a boundary.
fn fit_small<F, I>(data: &[F], max_bins: usize) -> Vec<F>
where
    F: QuantFloat,
    I: PrimInt + FromBytes + Immutable + IntoBytes,
{
    // The histogram has one slot per non-NaN bit pattern. `to_bits` is a
    // bijection from the non-NaN patterns onto a contiguous rank range, and
    // -infinity sits at its low end: its rank is `nan_size`, and there are
    // exactly `nan_size` NaN patterns on each side of the usable range (the
    // sign bit doubles them). Subtracting both halves leaves the width of
    // the usable range, so `bin = rank - nan_size` lands in `[0, hist_size)`.
    let nan_size = to_bits::<F, I>(F::neg_infinity()).to_usize().unwrap();
    let hist_size = (1 << (std::mem::size_of::<I>() * 8)) - nan_size * 2;

    // Calculate the histogram for non-NaN values.
    let mut hist = vec![0_usize; hist_size];
    let mut unique = 0;
    for value in data {
        if !value.is_nan() {
            let bin = to_bits::<F, I>(*value).to_usize().unwrap() - nan_size;
            unique += (hist[bin] == 0) as usize;
            hist[bin] += 1;
        }
    }
    trace!("histogram: {} unique values", unique);

    // Map a histogram slot back to its value.
    let convert = |bin: usize| from_bits::<F, I>(I::from(bin + nan_size).unwrap());
    if unique <= max_bins {
        // Exact: keep every distinct value, already monotone.
        return hist
            .into_iter()
            .enumerate()
            .filter(|(_, v)| *v != 0)
            .map(|(k, _)| convert(k))
            .collect::<Vec<_>>();
    }

    // Reduce the per-distinct-value counts to `max_bins` contiguous groups,
    // then emit the last value of each group as its boundary.
    let mut sizes = hist
        .iter()
        .filter_map(|&v| (v != 0).then_some(v))
        .collect::<Vec<_>>();
    sizes = merge_adjacent(&sizes, max_bins);

    // Walk the histogram in slot order. `rest` is how many non-zero slots
    // remain in the current group; when it hits zero the group's last slot
    // `k` is its boundary, so emit `convert(k)` and start the next group.
    let mut bins = Vec::with_capacity(max_bins);
    let mut iter = sizes.into_iter();
    let mut rest = iter.next().unwrap();
    for (k, v) in hist.into_iter().enumerate() {
        if v != 0 {
            rest -= v;
            if rest == 0 {
                bins.push(convert(k));
                rest = iter.next().unwrap_or(0);
            }
        }
    }
    bins
}

/// Sort-based fit for arbitrary-width floats.
///
/// Values are sorted with NaNs parked at the front (then dropped). The
/// first pass collects up to `max_bins` distinct values in order; if the
/// total distinct count is within `max_bins` that is the answer.
/// Otherwise it picks `max_bins * oversample` quantile cut points
/// (deduplicating repeats) and, if that still exceeds `max_bins`, merges
/// the smallest adjacent groups via `merge_adjacent`.
///
/// When several thresholds fall on the same value (heavily skewed input)
/// the repeat is dropped, so `bins` may be shorter than `max_bins`.
fn fit_large<F: QuantFloat>(data: &[F], max_bins: usize, oversample: usize) -> Vec<F> {
    // Total order with NaNs first, so `partition_point` can split them
    // off in a single sweep.
    let compare = |a: &F, b: &F| {
        if a.is_nan() || b.is_nan() {
            b.is_nan().cmp(&a.is_nan())
        } else {
            a.partial_cmp(b).unwrap()
        }
    };

    let mut data = data.to_vec();
    data.sort_unstable_by(compare);

    let start = data.partition_point(|x| x.is_nan());
    let slice = &data[start..];
    trace!("sort: {} total values", slice.len());
    if slice.is_empty() {
        return vec![];
    }

    // First pass: distinct values in sorted order (the first element plus
    // each change point), capped at `max_bins`.
    let bins = std::iter::once(slice[0])
        .chain(
            slice
                .array_windows()
                .filter_map(|[a, b]| (a != b).then_some(*b))
                .take(max_bins),
        )
        .collect::<Vec<_>>();
    if bins.len() <= max_bins {
        // Distinct values fit: `take` simply stopped after the last
        // one, so `bins` is the full distinct set.
        trace!("sort: {} unique values", bins.len());
        return bins;
    }

    // Collect the `samples` quantile cut points as row indices (each `pos`
    // is the 1-based end of a group), dropping consecutive thresholds that
    // land on the same value. If that is already within `max_bins`, the cut
    // points are the answer and we skip the merge entirely.
    let mut cuts: Vec<usize> = vec![];
    for pos in ThresholdIter::new(max_bins * oversample, slice.len()) {
        if cuts.last().is_none_or(|p| slice[p - 1] != slice[pos - 1]) {
            cuts.push(pos);
        }
    }
    if cuts.len() <= max_bins {
        trace!("sort: {} sampled values", cuts.len());
        return cuts.into_iter().map(|p| slice[p - 1]).collect();
    }

    // Each cut must land on a value boundary: advance cut `idx` to the last
    // row equal to the previous cut's value, so a group never straddles
    // distinct values. Then close the final group at the end of the slice.
    for idx in 1..cuts.len() {
        let [l, r] = cuts.get_disjoint_mut([idx - 1, idx]).unwrap();
        let value = slice[*l - 1];
        *l += slice[*l..*r].partition_point(|&x| x == value);
    }
    *cuts.last_mut().unwrap() = slice.len();

    // Convert boundaries to group sizes.
    let sizes = cuts
        .into_iter()
        .scan(0, |prev, pos| Some(pos - std::mem::replace(prev, pos)))
        .collect::<Vec<_>>();

    // Merge the groups and get the bin values.
    merge_adjacent(&sizes, max_bins)
        .into_iter()
        .scan(0, |pos, size| {
            *pos += size;
            Some(slice[*pos - 1])
        })
        .collect()
}

/// Greedily merge adjacent groups until exactly `target` remain, always
/// merging the smallest-sum adjacent pair first (a min-heap keyed on the
/// negative pair sum, so `BinaryHeap` pops it). Returns the size of each
/// surviving group, in group order.
///
/// `groups` is an intrusive linked list: each live group's `left`/`right`
/// point at its live neighbours, and a merged-away group is given
/// `size == 0` and immediately dereferenced by both neighbours (so it is
/// unreachable from the list). It then survives only as the index inside
/// stale heap entries, which the loop validates against the live groups
/// before applying.
fn merge_adjacent(sizes: &[usize], target: usize) -> Vec<usize> {
    trace!("merging {} groups to {}", sizes.len(), target);

    let mut count = sizes.len() - 1;
    let mut groups = sizes
        .iter()
        .enumerate()
        .map(|(i, n)| MergeGroup {
            size: *n as isize,
            left: (i != 0).then(|| i - 1),
            right: (i != count).then(|| i + 1),
        })
        .collect::<Vec<_>>();

    // One heap entry per adjacent pair, keyed on the (negative) pair sum so
    // the smallest pair is popped first. `i` is the left group of the pair.
    let mut heap = sizes
        .array_windows()
        .enumerate()
        .map(|(i, [l, r])| (-((*l + *r) as isize), i))
        .collect::<BinaryHeap<_>>();

    while count >= target {
        let (sum, idx) = heap.pop().unwrap();
        // Entries are applied to the right pair (`idx`, `idx`.right); the
        // stale check below rejects an entry whose pair no longer exists, so
        // `right` is present here.
        let adj = groups[idx].right.unwrap();
        let [g1, g2] = groups.get_disjoint_mut([idx, adj]).unwrap();
        // Stale: the recorded pair sum no longer matches the live groups, or
        // this group was already merged away.
        if g1.size + g2.size != -sum || g1.size == 0 {
            continue;
        }

        // Update the groups being merged.
        g1.size = -sum;
        g2.size = 0;
        g1.right = g2.right;

        // Re-key the pairs now adjacent to the merged group `idx`. The
        // neighbour sizes read here are the *new* merged size, matching what
        // a later stale check will compare against.
        let (left, right) = (g1.left, g2.right);
        if let Some(prev) = left {
            let g0 = &groups[prev];
            heap.push((sum - g0.size, prev));
        }
        if let Some(next) = right {
            let g3 = &mut groups[next];
            g3.left = Some(idx);
            heap.push((sum - g3.size, idx));
        }
        count -= 1;
    }

    // Walk the live groups (size > 0) via their `right` links and return each
    // group's total size as a contiguous block of the original `sizes`.
    let mut bounds = vec![];
    let mut next = Some(0);
    while let Some(idx) = next {
        bounds.push(idx);
        next = groups[idx].right;
    }
    bounds.push(sizes.len());
    refine_bounds(sizes, &mut bounds)
}

struct MergeGroup {
    size: isize,
    left: Option<usize>,
    right: Option<usize>,
}

/// Relax the internal boundaries of merged groups so each split is as
/// balanced as possible.
///
/// `ranges` is the bound list: `0`, then the internal split indices, then
/// `sizes.len()`. Each internal boundary lies between two neighbouring
/// groups; it is moved toward the sum midpoint of its span, and a move
/// changes the spans of the two adjacent boundaries, which are therefore
/// relaxed in turn (work queue + `dirty` set), until a fixed point is
/// reached. A boundary never moves on a tie: `b` is the first candidate
/// and `min_by_key` keeps the first minimum, so the result is stable.
///
/// Returns each group's total size, in group order.
fn refine_bounds(sizes: &[usize], ranges: &mut [usize]) -> Vec<usize> {
    // Prefix sums: `sums[k]` is the total of `sizes[..k]`.
    let sums = std::iter::once(0)
        .chain(sizes.iter().scan(0, |acc, s| {
            *acc += s;
            Some(*acc)
        }))
        .collect::<Vec<_>>();

    // Internal boundaries are the list indices `1..n`. `dirty` marks
    // boundaries already pending in the queue so they are not enqueued
    // twice.
    let n = ranges.len() - 1;
    let mut queue = VecDeque::from_iter(1..n);
    let mut dirty = vec![true; n - 1];
    let mut iter_count = 0;

    while let Some(i) = queue.pop_front() {
        dirty[i - 1] = false;
        iter_count += 1;

        // Boundary `b` splits the span between neighbours `a` and `c`;
        // relax it to the position that minimizes the left/right imbalance.
        let (a, b, c) = (ranges[i - 1], ranges[i], ranges[i + 1]);
        // Sum midpoint of the span (rounded up).
        let target = sums[a] + (sums[c] - sums[a]).div_ceil(2);
        // First boundary index reaching that midpoint. The result is
        // slice-relative, so add `a` back; it lies in `a..=c - 1`, which is
        // why the candidates are clamped below.
        let j = a + sums[a + 1..c].partition_point(|&v| v < target);

        // Left/right row-count imbalance around a candidate split.
        let imbalance = |k: &usize| {
            let (l, r) = (sums[*k] - sums[a], sums[c] - sums[*k]);
            l.abs_diff(r)
        };
        // Candidates: the current boundary plus the split positions closest
        // to the midpoint, clamped into the valid internal range
        // `a + 1..c - 1`.
        let best = [b, j.max(a + 1), (j + 1).min(c - 1)]
            .into_iter()
            .min_by_key(imbalance)
            .unwrap();

        if best != b {
            ranges[i] = best;
            // Moving this boundary shifts the spans of its neighbours, so
            // relax them again.
            for k in [i - 1, i + 1] {
                if k > 0 && k < n && !dirty[k - 1] {
                    queue.push_back(k);
                    dirty[k - 1] = true;
                }
            }
        }
    }
    trace!("refinement iterations: {}", iter_count);

    // Map each window to its group's total size.
    ranges
        .array_windows()
        .map(|[l, r]| sums[*r] - sums[*l])
        .collect()
}

/// Quantile threshold generator.
///
/// Yields `min(max_bins, total)` non-decreasing row counts in `[0..=total]`:
/// the `k`-th one is `round(total * k / max_bins)` for `k = 1..=max_bins`.
struct ThresholdIter {
    bin: usize,
    max_bins: usize,
    total: usize,
}

impl ThresholdIter {
    fn new(max_bins: usize, total: usize) -> Self {
        Self {
            bin: 0,
            max_bins: max_bins.min(total),
            total,
        }
    }
}

impl Iterator for ThresholdIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.bin >= self.max_bins {
            return None;
        }
        self.bin += 1;
        // Exact integer round of `total * bin / max_bins`: adding half the
        // denominator before truncating is equivalent to rounding to nearest.
        let num = 2 * self.bin * self.total + self.max_bins;
        let den = 2 * self.max_bins;
        Some(num / den)
    }
}

/// Map a float to a monotone unsigned bit rank: `a < b` implies
/// `to_bits(a) < to_bits(b)` (with `-0.0` and `0.0` sharing a rank).
/// The IEEE sign bit is reinterpreted as a half marker: negative values
/// (sign bit set) map to the low half with the result sign bit cleared,
/// positives (sign bit clear) map to the high half with it set. Within
/// each half the magnitude bits are kept monotonically ordered, so a
/// plain integer comparison orders the values correctly.
fn to_bits<F, I>(value: F) -> I
where
    F: Float + IntoBytes + Immutable,
    I: PrimInt + FromBytes,
{
    let bits = I::read_from_bytes(value.as_bytes()).unwrap();
    let sign = I::one() << (std::mem::size_of::<I>() * 8 - 1);
    if bits == sign {
        return bits; // T::NEG_ZERO is equal to T::ZERO
    }
    if bits & sign == sign {
        !bits
    } else {
        bits ^ sign
    }
}

/// Inverse of `to_bits`: map a monotone unsigned bit rank back to its
/// float. The round trip is `from_bits(to_bits(x)) == x` for every valid
/// float `x` (value equality: `-0.0` round-trips to `0.0`).
fn from_bits<F, I>(bits: I) -> F
where
    F: Float + FromBytes,
    I: PrimInt + IntoBytes + Immutable,
{
    let sign = I::one() << (std::mem::size_of::<I>() * 8 - 1);
    let value = if bits & sign != sign {
        !bits
    } else {
        bits ^ sign
    };
    F::read_from_bytes(value.as_bytes()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn quantize() {
        let bins = vec![1.0, 2.0, 3.0];
        let quantize = |x: f32| super::quantize::<f32, u8>(&x, &bins);
        assert_eq!(quantize(f32::NAN), 0);
        assert_eq!(quantize(f32::NEG_INFINITY), 1);
        assert_eq!(quantize(1.0), 1);
        assert_eq!(quantize(1.5), 2);
        assert_eq!(quantize(2.0), 2);
        assert_eq!(quantize(2.5), 3);
        assert_eq!(quantize(3.0), 3);
        assert_eq!(quantize(f32::INFINITY), 3);
    }

    #[test]
    fn quantize_all() {
        let col1 = [f32::NAN, 1.0, 1.5, 2.0, 2.5, 3.0, f32::INFINITY];
        let col2 = [f32::NEG_INFINITY, -1.0, 0.0, 1.0];
        let data: Vec<&[f32]> = vec![&col1, &col2];

        for parallel in [false, true] {
            let bins = bin_all(&data, 3, 1, parallel);
            let pairs = data
                .iter()
                .zip(bins.iter())
                .map(|(a, b)| (*a, b.as_slice()))
                .collect::<Vec<_>>();
            let actual = super::quantize_all::<f32, u8>(&pairs, parallel);
            assert_eq!(actual[0], vec![0, 1, 1, 2, 2, 3, 3]);
            assert_eq!(bins[0], vec![1.5, 2.5, f32::INFINITY]);
            assert_eq!(actual[1], vec![1, 2, 2, 3]);
            assert_eq!(bins[1], vec![f32::NEG_INFINITY, 0.0, 1.0]);
        }
    }

    #[test]
    fn empty_data() {
        let data_f16 = Vec::<f16>::new();
        assert_eq!(fit_small::<f16, u16>(&data_f16, 2), vec![]);

        let data_f32 = Vec::<f32>::new();
        assert_eq!(fit_large(&data_f32, 2, 1), Vec::<f32>::new());
    }

    #[test]
    fn all_values_nan() {
        let data_f16 = vec![f16::NAN; 10];
        assert_eq!(fit_small::<f16, u16>(&data_f16, 2), vec![]);

        let data_f32 = vec![f32::NAN; 10];
        assert_eq!(fit_large(&data_f32, 2, 1), Vec::<f32>::new());
    }

    #[test]
    fn unique_balanced() {
        let data_f16 = vec![
            f16::NEG_INFINITY,
            f16::NEG_ONE,
            f16::ZERO,
            f16::ONE,
            f16::INFINITY,
        ];
        assert_eq!(fit_small::<f16, u16>(&data_f16, 5), data_f16);

        let data_f32 = vec![f32::NEG_INFINITY, -1.0, 0.0, 1.0, f32::INFINITY];
        assert_eq!(fit_large(&data_f32, 5, 1), data_f32);
    }

    #[test]
    fn unique_imbalanced() {
        let data_f16 = [vec![f16::ZERO; 10], vec![f16::ONE, f16::NEG_ONE]].concat();
        assert_eq!(
            fit_small::<f16, u16>(&data_f16, 4),
            vec![f16::NEG_ONE, f16::ZERO, f16::ONE]
        );

        let data_f32 = [vec![0.0; 10], vec![1.0, -1.0]].concat();
        assert_eq!(fit_large(&data_f32, 4, 1), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn sample_balanced() {
        let c = |x: usize| f16::from_f32(x as f32);
        let data_f16 = (0..100).map(|i| c(i % 10)).collect::<Vec<_>>();
        assert_eq!(
            fit_small::<f16, u16>(&data_f16, 5),
            vec![c(1), c(3), c(5), c(7), c(9)]
        );

        let data_f32 = (0..100).map(|i| (i % 10) as f32).collect::<Vec<_>>();
        assert_eq!(fit_large(&data_f32, 5, 1), vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn sample_imbalanced() {
        let data_f16 = [
            vec![f16::ZERO; 8],
            vec![f16::ONE, f16::NEG_ONE, f16::MIN, f16::MAX],
        ]
        .concat();
        assert_eq!(
            fit_small::<f16, u16>(&data_f16, 3),
            vec![f16::NEG_ONE, f16::ZERO, f16::MAX]
        );

        let data_f32 = [vec![0.0; 8], vec![1.0, -1.0, f32::MIN, f32::MAX]].concat();
        assert_eq!(fit_large(&data_f32, 3, 2), vec![-1.0, 0.0, f32::MAX]);
        assert_eq!(fit_large(&data_f32, 3, 1), vec![0.0, f32::MAX]);
    }

    #[test]
    fn sample_imbalanced_nan() {
        let data_f16 = [
            vec![f16::NAN; 4],
            vec![f16::ONE, f16::NEG_ONE, f16::ZERO, f16::NEG_ZERO],
        ]
        .concat();
        assert_eq!(
            fit_small::<f16, u16>(&data_f16, 2),
            vec![f16::NEG_ONE, f16::ONE]
        );

        let data_f32 = [vec![f32::NAN; 4], vec![1.0, -1.0, 0.0, -0.0]].concat();
        assert_eq!(fit_large(&data_f32, 2, 2), vec![-1.0, 1.0]);
    }

    #[test]
    fn fit_from_empty() {
        let data_f16 = Vec::<f16>::new();
        assert_eq!(bins_from(&data_f16, 2, 1), vec![f16::INFINITY]);

        let data_f32 = Vec::<f32>::new();
        assert_eq!(bins_from(&data_f32, 2, 1), vec![f32::INFINITY]);
    }

    #[test]
    fn fit_from_large() {
        let data_f16 = [f16::MIN, f16::ZERO, f16::ONE, f16::MAX].repeat(1024);
        assert_eq!(
            bins_from(&data_f16, 3, 1),
            vec![f16::MIN, f16::ZERO, f16::MAX]
        );

        let data_f32 = [f32::MIN, 0.0, 1.0, f32::MAX].repeat(1024);
        assert_eq!(bins_from(&data_f32, 3, 2), vec![f32::MIN, 0.0, f32::MAX]);
    }

    #[test]
    fn merge_heap() {
        assert_eq!(merge_adjacent(&[1, 5, 3, 5, 100], 3), [6, 8, 100]);
        assert_eq!(merge_adjacent(&[1, 5, 3, 5, 100], 2), [14, 100]);
        assert_eq!(merge_adjacent(&[8, 1, 1, 8], 2), [9, 9]);
    }
}
