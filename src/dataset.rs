//! Training data model: feature descriptors, the row/column dataset,
//! per-row loss derivatives, and the objective functions they are driven by.
//!
//! Structures:
//! - `FeatureType`: how a bin is matched (exact or at-or-below threshold).
//! - `Feature`: one column's name, type, and bin boundaries.
//! - `Dataset`: the feature matrix and target values.
//! - `RowIndex`: one row's index plus its gradient and hessian.
//! - `IndexT`: integer index trait (convertible to and from `usize`).
//! - `Objective`: loss and derivative provider trait.
//! - `SquaredLoss`: regression objective (squared error).
//! - `LogLoss`: classification objective (binary cross-entropy).

/// Feature type defines how values are matched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeatureType {
    /// Match exactly one bin (`==`).
    OneHot,
    /// Match the bin and any bin below it (`<=`).
    Linear,
}

/// Feature descriptor defines the name, type and original domain thresholds.
#[derive(Clone, Debug)]
pub struct Feature {
    /// Original column name from the source data.
    pub name: String,
    /// Type defines how values are matched.
    pub type_: FeatureType,
    /// The bin boundaries split the original domain into `bins.len()` ranges.
    /// The last bin contains the maximum known value, but is treated as infinity.
    /// Values [1, 2, 3] define ranges (-inf, 1]; (1, 2]; (2, inf).
    pub bins: Vec<f32>,
}

impl Feature {
    /// Build a feature with `FeatureType::OneHot`.
    pub fn onehot(name: String, bins: Vec<f32>) -> Self {
        Self {
            type_: FeatureType::OneHot,
            name,
            bins,
        }
    }

    /// Build a feature with `FeatureType::Linear`.
    pub fn linear(name: String, bins: Vec<f32>) -> Self {
        Self {
            type_: FeatureType::Linear,
            name,
            bins,
        }
    }
}

/// A linear feature with no name and no bins (used in tests).
impl Default for Feature {
    fn default() -> Self {
        Self::linear(String::new(), Vec::new())
    }
}

/// Training data: feature matrix (X) and target values (y).
#[derive(Clone)]
pub struct Dataset<T> {
    /// One descriptor per feature column.
    pub features: Vec<Feature>,
    /// One inner vector per feature, one value per row.
    pub inputs: Vec<Vec<T>>,
    /// Per-row target value (classification label or regression value).
    pub target: Vec<f32>,
}

impl<T> Dataset<T> {
    /// Build a new dataset from its parts.
    pub fn new(features: Vec<Feature>, inputs: Vec<Vec<T>>, target: Vec<f32>) -> Self {
        Self {
            features,
            inputs,
            target,
        }
    }

    /// Number of columns in the dataset.
    pub fn columns(&self) -> usize {
        self.inputs.len()
    }

    /// Number of rows in the dataset.
    pub fn rows(&self) -> usize {
        self.inputs.first().map_or(0, |x| x.len())
    }

    /// Check that the structure sizes agree.
    pub fn is_valid(&self) -> bool {
        self.features.len() == self.columns()
            && self.inputs.iter().all(|col| col.len() == self.rows())
            && (self.target.is_empty() || self.target.len() == self.rows())
    }
}

/// A single row of the dataset with the per-iteration loss derivatives.
#[derive(Clone, Copy)]
pub struct RowIndex<I: IndexT, O: Objective> {
    /// Row position, dataset-wide or sampled-subset-relative.
    pub index: I,
    /// Stored gradient/hessian values.
    pub deriv: O,
}

impl<I: IndexT, O: Objective> RowIndex<I, O> {
    /// Constructor; `index` should be validated by the user.
    pub fn new(index: usize, deriv: O) -> Self {
        RowIndex {
            index: I::from_usize(index),
            deriv,
        }
    }

    /// Construct new empty index vector.
    pub fn new_array(size: usize) -> Vec<Self> {
        (0..size)
            .map(|idx| RowIndex::new(idx, O::derive(0.0, 0.0)))
            .collect()
    }
}

/// Integer index trait (convertible to and from `usize`).
pub trait IndexT: Copy + Send + Sync {
    fn from_usize(index: usize) -> Self;
    fn to_usize(&self) -> usize;
}

impl IndexT for u64 {
    fn from_usize(index: usize) -> Self {
        index as u64
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl IndexT for u32 {
    fn from_usize(index: usize) -> Self {
        index as u32
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

/// A loss function used to drive gradient boosting.
pub trait Objective: Copy + Send + Sync {
    /// The loss of a predicted value against the true label.
    fn loss(y: f32, y_pred: f32) -> f32;

    /// Derive gradient/hessian values of the loss.
    fn derive(y: f32, y_pred: f32) -> Self;

    /// Return stored gradient/hessian values.
    fn grad_hess(&self) -> (f32, f32);
}

/// Regression loss: squared error.
#[derive(Clone, Copy)]
pub struct SquaredLoss(f32);

impl Objective for SquaredLoss {
    /// Loss: (y^ - y) ^ 2 / 2
    fn loss(y: f32, y_pred: f32) -> f32 {
        let d = y_pred - y;
        0.5 * d * d
    }

    /// Gradient: y^ - y; Hessian: 1.0
    fn derive(y: f32, y_pred: f32) -> Self {
        Self(y_pred - y)
    }

    fn grad_hess(&self) -> (f32, f32) {
        (self.0, 1.0)
    }
}

/// Classification loss: binary cross-entropy against `y in {0, 1}`.
#[derive(Clone, Copy)]
pub struct LogLoss(f32, f32);

impl Objective for LogLoss {
    /// Loss: -(y * log(y^) + (1 - y) * log(1 - y^))
    fn loss(y: f32, y_pred: f32) -> f32 {
        let y_hat = odds(y_pred);
        -(y * y_hat.ln() + (1.0 - y) * (1.0 - y_hat).ln())
    }

    /// Gradient: y^ - y; Hessian: y^ * (1 - y^)
    fn derive(y: f32, y_pred: f32) -> Self {
        let y_hat = odds(y_pred);
        Self(y_hat - y, y_hat * (1.0 - y_hat))
    }

    fn grad_hess(&self) -> (f32, f32) {
        (self.0, self.1)
    }
}

/// The sigmoid of `y`: maps a raw score to a probability in (0, 1).
fn odds(y: f32) -> f32 {
    1.0 / (1.0 + (-y).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_small() {
        // Small dataset (`u8` bins, `u32` index).
        let features = vec![Feature::default(); 2];
        let input = vec![vec![0, 1, 2]; 2];
        let data = Dataset::<u8>::new(features, input, vec![1.0; 3]);

        assert!(data.is_valid());
        assert_eq!(data.rows(), 3);
        assert_eq!(data.columns(), 2);
    }

    #[test]
    fn valid_large() {
        // Large dataset (`u16` bins, `u64` index, target offloaded).
        let features = vec![Feature::default(); 2];
        let input = vec![vec![0; 1000]; 2];
        let data = Dataset::<u16>::new(features.clone(), input, vec![]);

        assert!(data.is_valid());
        assert_eq!(data.rows(), 1000);
        assert_eq!(data.columns(), 2);
    }

    #[test]
    fn invalid_data() {
        let features = vec![Feature::default(); 2];
        let input = vec![vec![0; 3]; 2];
        let create = |f, x, y| Dataset::<u8>::new(f, x, y);

        let missing_features = create(vec![], input.clone(), vec![]);
        assert!(!missing_features.is_valid());
        let invalid_input = create(features.clone(), vec![vec![0; 3], vec![1]], vec![]);
        assert!(!invalid_input.is_valid());
        let invalid_target = create(features.clone(), input.clone(), vec![1.0]);
        assert!(!invalid_target.is_valid());
    }

    #[test]
    fn squared_loss() {
        assert_eq!(SquaredLoss::loss(1.0, 4.0), 4.5);
        assert_eq!(SquaredLoss::derive(1.0, 4.0).grad_hess(), (3.0, 1.0));
    }

    #[test]
    fn log_loss() {
        assert_eq!(LogLoss::loss(0.5, 0.75), 0.76187104);
        assert_eq!(
            LogLoss::derive(0.5, 0.75).grad_hess(),
            (0.17917871, 0.21789499)
        );
    }
}
