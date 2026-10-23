//! Training data model: feature descriptors, the row/column dataset,
//! per-row loss derivatives, and the objective functions they are driven by.
//!
//! Structures:
//! - `FeatureType`: how a bin is matched (exact or at-or-below threshold).
//! - `Feature`: one column's name, type, and bin boundaries.
//! - `Dataset`: the feature matrix, target, predictions, and row gradients.
//! - `RowIndexT`: integer row-index trait (convertible to and from `usize`).
//! - `RowIndex`: one row's index plus its gradient and hessian.
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

/// Training data: feature matrix, target, running predictions, and row index.
#[derive(Clone, Debug)]
pub struct Dataset<T, I: RowIndexT> {
    /// One descriptor per feature column.
    pub features: Vec<Feature>,
    /// One inner vector per feature, one value per row.
    pub inputs: Vec<Vec<T>>,
    /// Per-row target value (classification label or regression value).
    pub target: Vec<f32>,
    /// Per-row running predictions accumulated across boosting rounds.
    pub predict: Vec<f32>,
    /// One entry per row; carries the gradient/hessian for this iteration.
    pub index: Vec<RowIndex<I>>,
}

impl<T, I: RowIndexT> Dataset<T, I> {
    /// Build a new dataset from its parts. `predict` is seeded to `initial`
    /// for every target, and `index` is initialized to zero derivative.
    pub fn new(
        features: Vec<Feature>,
        inputs: Vec<Vec<T>>,
        target: Vec<f32>,
        initial: f32,
    ) -> Self {
        let rows = inputs.first().map(|x| x.len()).unwrap_or(0);
        let index = (0..rows).map(|i| RowIndex::new(i, 0.0, 0.0)).collect();
        let predict = vec![initial; target.len()];
        Self {
            features,
            inputs,
            target,
            predict,
            index,
        }
    }

    /// Number of columns in the dataset.
    pub fn columns(&self) -> usize {
        self.inputs.len()
    }

    /// Number of rows in the dataset.
    pub fn rows(&self) -> usize {
        self.inputs.first().map(|x| x.len()).unwrap_or(0)
    }

    /// Check that the structure sizes agree.
    pub fn is_valid(&self) -> bool {
        self.features.len() == self.columns()
            && self.inputs.iter().all(|col| col.len() == self.rows())
            && (self.target.is_empty() || self.target.len() == self.rows())
            && self.predict.len() == self.target.len()
            && self.index.len() == self.rows()
    }
}

/// Integer row-index trait (convertible to and from `usize`).
pub trait RowIndexT: Copy + Send {
    fn from_usize(index: usize) -> Self;
    fn to_usize(&self) -> usize;
}

impl RowIndexT for u64 {
    fn from_usize(index: usize) -> Self {
        index as u64
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

impl RowIndexT for u32 {
    fn from_usize(index: usize) -> Self {
        index as u32
    }

    fn to_usize(&self) -> usize {
        *self as usize
    }
}

/// A single row of the dataset plus the per-iteration loss derivatives.
#[derive(Clone, Copy, Debug)]
pub struct RowIndex<T: RowIndexT> {
    /// Row position in the dataset.
    pub index: T,
    /// First derivative of the loss for this row.
    pub gradient: f32,
    /// Second derivative of the loss for this row.
    pub hessian: f32,
}

impl<T: RowIndexT> RowIndex<T> {
    /// Constructor; `index` should be validated by the user.
    pub fn new(index: usize, gradient: f32, hessian: f32) -> Self {
        RowIndex {
            index: T::from_usize(index),
            gradient,
            hessian,
        }
    }
}

/// A loss function used to drive gradient boosting.
pub trait Objective {
    /// The loss of a predicted value against the true label.
    fn loss(y: f32, y_pred: f32) -> f32;

    /// The first and second derivatives of the loss with respect to `y_pred`.
    fn grad_hess(y: f32, y_pred: f32) -> (f32, f32);
}

/// Regression loss: squared error.
pub struct SquaredLoss;

impl Objective for SquaredLoss {
    /// Loss: (y^ - y) ^ 2 / 2
    fn loss(y: f32, y_pred: f32) -> f32 {
        let d = y_pred - y;
        0.5 * d * d
    }

    /// Gradient: y^ - y; Hessian: 1.0
    fn grad_hess(y: f32, y_pred: f32) -> (f32, f32) {
        (y_pred - y, 1.0)
    }
}

/// Classification loss: binary cross-entropy against `y in {0, 1}`.
pub struct LogLoss;

impl Objective for LogLoss {
    /// Loss: -(y * log(y^) + (1 - y) * log(1 - y^))
    fn loss(y: f32, y_pred: f32) -> f32 {
        let y_hat = odds(y_pred);
        -(y * y_hat.ln() + (1.0 - y) * (1.0 - y_hat).ln())
    }

    /// Gradient: y^ - y; Hessian: y^ * (1 - y^)
    fn grad_hess(y: f32, y_pred: f32) -> (f32, f32) {
        let y_hat = odds(y_pred);
        (y_hat - y, y_hat * (1.0 - y_hat))
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
        let data = Dataset::<u8, u32>::new(features, input, vec![1.0; 3], 0.5);

        assert!(data.is_valid());
        assert_eq!(data.rows(), 3);
        assert_eq!(data.columns(), 2);
        assert_eq!(data.predict, vec![0.5; 3]);
    }

    #[test]
    fn valid_large() {
        // Large dataset (`u16` bins, `u64` index, target/predict offloaded).
        let features = vec![Feature::default(); 2];
        let input = vec![vec![0; 1000]; 2];
        let data = Dataset::<u16, u64>::new(features.clone(), input, vec![], 0.0);

        assert!(data.is_valid());
        assert_eq!(data.rows(), 1000);
        assert_eq!(data.columns(), 2);
        assert!(data.predict.is_empty());
    }

    #[test]
    fn invalid_data() {
        let features = vec![Feature::default(); 2];
        let input = vec![vec![0; 3]; 2];
        let create = |f, x, y| Dataset::<u8, u32>::new(f, x, y, 0.0);

        let missing_features = create(vec![], input.clone(), vec![]);
        assert!(!missing_features.is_valid());
        let invalid_input = create(features.clone(), vec![vec![0; 3]], vec![]);
        assert!(!invalid_input.is_valid());
        let invalid_target = create(features.clone(), input.clone(), vec![1.0]);
        assert!(!invalid_target.is_valid());
    }

    #[test]
    fn squared_loss() {
        assert_eq!(SquaredLoss::loss(1.0, 4.0), 4.5);
        assert_eq!(SquaredLoss::grad_hess(1.0, 4.0), (3.0, 1.0));
    }

    #[test]
    fn log_loss() {
        assert_eq!(LogLoss::loss(0.5, 0.75), 0.76187104);
        assert_eq!(LogLoss::grad_hess(0.5, 0.75), (0.17917871, 0.21789499));
    }
}
