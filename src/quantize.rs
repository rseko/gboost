//! Bin-boundary fitting: turn a feature's raw values into the sorted
//! thresholds the tree compares against, and map each value to its
//! 1-based bin index among those thresholds.
//!
//! - `quantize` maps a single value to its bin index; NaN maps to 0.
//! - `quantize_all` fits every column and quantizes it in one pass,
//!   returning `(quantized, bins)` per column.
//! - `bins_from` is the fitting entry point; NaNs are always dropped,
//!   and an empty result is normalized to a single positive infinity.
//!   - `fit_small` uses an exact float16 histogram (total ordering) and
//!     quantile-walks the histogram to choose the bins.
//!   - `fit_large` sorts the values and either keeps the distinct values
//!     or samples them evenly by rank.
//! - `QuantFloat` is the bound every public float in this module shares.

use num_traits::{Float, PrimInt};
use rayon::prelude::*;
use zerocopy::{FromBytes, Immutable, IntoBytes};

/// Map a value to its 1-based bin index among the sorted `bins`.
/// With `bins = [1, 2, 3]`: `value <= 1` → 1, `1 < value <= 2` → 2,
/// `value > 2` → 3. NaN maps to 0.
pub fn quantize<F: Float, I: PrimInt>(value: &F, bins: &[F]) -> I {
    if !value.is_nan() {
        let pos = bins.partition_point(|x| value > x) + 1;
        I::from(pos.min(bins.len())).unwrap_or(I::zero())
    } else {
        I::zero()
    }
}

/// A floating-point type suitable for quantization.
pub trait QuantFloat: Float + FromBytes + Immutable + IntoBytes + Send + Sync {}
impl<T: Float + FromBytes + Immutable + IntoBytes + Send + Sync> QuantFloat for T {}

/// Fit a quantizer per feature column and map each value to its bin
/// index. Returns `(quantized, bins)`, both in column order; `quantized`
/// is one `Vec<I>` per column (1-based, NaN -> 0), `bins` the fitted
/// boundaries that produced them.
pub fn quantize_all<F: QuantFloat, I: PrimInt + Send>(
    data: &[&[F]],
    max_bins: usize,
    parallel: bool,
) -> (Vec<Vec<I>>, Vec<Vec<F>>) {
    let quantize = |row: &&[F]| {
        let bins = bins_from(row, max_bins);
        let data = row.iter().map(|x| quantize(x, &bins)).collect();
        (data, bins)
    };
    if parallel {
        data.par_iter().map(quantize).unzip()
    } else {
        data.iter().map(quantize).unzip()
    }
}

/// Fit up to `max_bins` boundaries from `data`.
///
/// NaNs are never treated as values. Dispatches to the histogram
/// algorithm for float16 on large batches, otherwise to the sort-based
/// one. The returned `bins` is always non-empty.
pub fn bins_from<F: QuantFloat>(data: &[F], max_bins: usize) -> Vec<F> {
    assert!(max_bins > 0);

    let n_bytes = std::mem::size_of::<F>();
    // The small path allocates a 64k histogram; below this many values
    // sorting is cheaper, hence the size gate alongside the type gate.
    let mut bins = if n_bytes == 2 && data.len() >= 4096 {
        fit_small::<_, u16>(data, max_bins)
    } else {
        fit_large(data, max_bins)
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
/// value is kept; otherwise quantile thresholds are walked in slot
/// order and one boundary per threshold is emitted.
///
/// When several thresholds land in the same histogram slot (heavily
/// skewed input) the repeated boundary is dropped, so `bins` may end
/// up shorter than `max_bins`.
fn fit_small<F, I>(data: &[F], max_bins: usize) -> Vec<F>
where
    F: QuantFloat,
    I: PrimInt + FromBytes + Immutable + IntoBytes,
{
    // Monotone rank of -infinity: all values >= -inf sit above it, and
    // negative NaN values below it (same number as positive NaN values),
    // so the usable range is exactly the histogram width below.
    let nan_size = to_bits::<F, I>(F::neg_infinity()).to_usize().unwrap();
    let hist_size = (1 << (std::mem::size_of::<I>() * 8)) - nan_size * 2;

    // Calculate the histogram and count NaN values.
    let mut hist = vec![0_usize; hist_size];
    let mut unique = 0;
    let mut nan_count = 0;
    for value in data {
        if !value.is_nan() {
            let bin = to_bits::<F, I>(*value).to_usize().unwrap() - nan_size;
            unique += (hist[bin] == 0) as usize;
            hist[bin] += 1;
        } else {
            nan_count += 1;
        }
    }

    // Map a histogram slot back to its value.
    let convert = |bin: usize| from_bits::<F, I>(I::from(bin + nan_size).unwrap());
    if unique <= max_bins {
        // Exact: keep every distinct value, already monotone.
        return hist
            .into_iter()
            .enumerate()
            .filter_map(|(k, v)| if v != 0 { Some(convert(k)) } else { None })
            .collect::<Vec<_>>();
    }

    // Quantile walk: the thresholds index the sorted (NaN-free) rows.
    // `hist` is sorted by `to_bits`, so scanning slots accumulates the
    // same counts in order.
    let mut bins = Vec::with_capacity(max_bins);
    let mut prev: Option<usize> = None;
    let (mut idx, mut acc) = (0, 0);
    for pos in ThresholdIter::new(max_bins, data.len() - nan_count) {
        // Advance to the slot whose cumulative count first reaches the
        // threshold; `acc` includes that slot once advanced.
        while acc < pos {
            acc += hist[idx];
            idx += 1;
        }
        // `idx - 1` is the last slot at or below the threshold.
        // Skip a slot already emitted for a previous threshold.
        if prev.is_none_or(|x| x != idx) {
            prev = Some(idx);
            bins.push(convert(idx - 1));
        }
    }
    bins
}

/// Sort-based fit for arbitrary-width floats.
///
/// Values are sorted with NaNs parked at the front (then dropped). The
/// first pass collects the distinct values in order; if that is within
/// `max_bins` it is the answer. Otherwise values are sampled at the
/// same quantile thresholds as `fit_small`, one boundary each.
///
/// When several thresholds fall on the same value (heavily skewed input)
/// the repeat is dropped, so `bins` may be shorter than `max_bins`.
fn fit_large<F: QuantFloat>(data: &[F], max_bins: usize) -> Vec<F> {
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
    if slice.is_empty() {
        return vec![];
    }

    // Pass 1: distinct values in sorted order (the first element plus
    // each change point), capped at `max_bins`.
    let mut bins = std::iter::once(slice[0])
        .chain(
            slice
                .array_windows()
                .filter_map(|[a, b]| if a != b { Some(*b) } else { None })
                .take(max_bins),
        )
        .collect::<Vec<_>>();
    if bins.len() <= max_bins {
        // Distinct values fit: `take` simply stopped after the last
        // one, so `bins` is the full distinct set.
        return bins;
    }

    // Pass 2: more distinct values than `max_bins`, sample by rank at
    // the quantile thresholds and drop consecutive repeats.
    bins.clear();
    let mut prev: Option<F> = None;
    for pos in ThresholdIter::new(max_bins, slice.len()) {
        let cur = slice[pos - 1];
        if prev.is_none_or(|x| x != cur) {
            prev = Some(cur);
            bins.push(cur);
        }
    }
    bins
}

/// Quantile threshold generator.
///
/// Yields `max_bins` non-decreasing row counts in `[0..=total]`: the
/// `k`-th one is `round(total * k / max_bins)` for `k = 1..=max_bins`.
struct ThresholdIter {
    bin: usize,
    max_bins: usize,
    total: usize,
}

impl ThresholdIter {
    fn new(max_bins: usize, total: usize) -> Self {
        assert!(max_bins <= total);
        Self {
            bin: 0,
            max_bins,
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
        let part = self.bin as f64 / self.max_bins as f64;
        Some((self.total as f64 * part).round() as usize)
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
            let (actual, bins) = super::quantize_all::<f32, u8>(&data, 3, parallel);
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
        assert_eq!(fit_large(&data_f32, 2), Vec::<f32>::new());
    }

    #[test]
    fn all_values_nan() {
        let data_f16 = vec![f16::NAN; 10];
        assert_eq!(fit_small::<f16, u16>(&data_f16, 2), vec![]);

        let data_f32 = vec![f32::NAN; 10];
        assert_eq!(fit_large(&data_f32, 2), Vec::<f32>::new());
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
        assert_eq!(fit_large(&data_f32, 5), data_f32);
    }

    #[test]
    fn unique_imbalanced() {
        let data_f16 = [vec![f16::ZERO; 10], vec![f16::ONE, f16::NEG_ONE]].concat();
        assert_eq!(
            fit_small::<f16, u16>(&data_f16, 4),
            vec![f16::NEG_ONE, f16::ZERO, f16::ONE]
        );

        let data_f32 = [vec![0.0; 10], vec![1.0, -1.0]].concat();
        assert_eq!(fit_large(&data_f32, 4), vec![-1.0, 0.0, 1.0]);
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
        assert_eq!(fit_large(&data_f32, 5), vec![1.0, 3.0, 5.0, 7.0, 9.0]);
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
            vec![f16::ZERO, f16::MAX]
        );

        let data_f32 = [vec![0.0; 8], vec![1.0, -1.0, f32::MIN, f32::MAX]].concat();
        assert_eq!(fit_large(&data_f32, 3), vec![0.0, f32::MAX]);
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
            vec![f16::ZERO, f16::ONE]
        );

        let data_f32 = [vec![f32::NAN; 4], vec![1.0, -1.0, 0.0, -0.0]].concat();
        assert_eq!(fit_large(&data_f32, 2), vec![0.0, 1.0]);
    }

    #[test]
    fn fit_from_empty() {
        let data_f16 = Vec::<f16>::new();
        assert_eq!(bins_from(&data_f16, 2), vec![f16::INFINITY]);

        let data_f32 = Vec::<f32>::new();
        assert_eq!(bins_from(&data_f32, 2), vec![f32::INFINITY]);
    }

    #[test]
    fn fit_from_large() {
        let data_f16 = [f16::MIN, f16::ZERO, f16::ONE, f16::MAX].repeat(1024);
        assert_eq!(bins_from(&data_f16, 3), vec![f16::ZERO, f16::ONE, f16::MAX]);

        let data_f32 = [f32::MIN, 0.0, 1.0, f32::MAX].repeat(1024);
        assert_eq!(bins_from(&data_f32, 3), vec![0.0, 1.0, f32::MAX]);
    }
}
