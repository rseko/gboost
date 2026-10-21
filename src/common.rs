//! Common definitions shared across the gradient boosting library.

/// Unit size of a data block processed by most internal operations.
pub const BSIZE: usize = 256;
/// A fixed-size stack-allocated block of data, sized to `BSIZE`.
pub type Batch<T> = [T; BSIZE];

/// An integer row-index width: convertible to and from `usize`.
pub trait Index: TryFrom<usize> + Copy {
    fn get(&self) -> usize;
}

impl Index for u64 {
    #[inline]
    fn get(&self) -> usize {
        *self as usize
    }
}

impl Index for u32 {
    #[inline]
    fn get(&self) -> usize {
        *self as usize
    }
}
