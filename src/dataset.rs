//! Training data model: feature descriptors, the row/column dataset,
//! per-row loss derivatives, and the objective functions they are driven by.
//!
//! Structures:
//! - `FeatureKind`: how a bin is matched (exact or at-or-below threshold).
//! - `Feature`: one column's name, kind, and bin boundaries.
//! - `Dataset`: the feature matrix and target values.
//! - `RowIndex`: one row's index plus its gradient and hessian.
//! - `IndexT`: integer index trait (convertible to and from `usize`).
//! - `Objective`: loss and derivative provider trait.
//! - `SquaredLoss`: regression objective (squared error).
//! - `LogLoss`: classification objective (binary cross-entropy).

use anyhow::{Result, bail};

/// Feature kind defines how values are matched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeatureKind {
    /// Match exactly one bin (`==`).
    OneHot,
    /// Match the bin and any bin below it (`<=`).
    Linear,
}

/// Feature descriptor defines the name, kind and original domain thresholds.
#[derive(Clone, Debug)]
pub struct Feature {
    /// Original column name from the source data.
    name: String,
    /// Kind defines how values are matched.
    kind: FeatureKind,
    /// The bin boundaries split the original domain into `bins.len()` ranges.
    /// The last bin contains the maximum known value, but is treated as infinity.
    /// Values [1, 2, 3] define ranges (-inf, 1]; (1, 2]; (2, inf).
    bins: Vec<f32>,
}

impl Feature {
    /// Build a feature with `FeatureKind::OneHot`.
    pub fn onehot(name: String, bins: Vec<f32>) -> Self {
        Self {
            kind: FeatureKind::OneHot,
            name,
            bins,
        }
    }

    /// Build a feature with `FeatureKind::Linear`.
    pub fn linear(name: String, bins: Vec<f32>) -> Self {
        Self {
            kind: FeatureKind::Linear,
            name,
            bins,
        }
    }

    /// Get feature name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get feature kind.
    pub fn kind(&self) -> FeatureKind {
        self.kind
    }

    /// Get bin boundaries.
    pub fn bins(&self) -> &[f32] {
        &self.bins
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
    features: Vec<Feature>,
    /// One inner vector per feature, one value per row.
    inputs: Vec<Vec<T>>,
    /// Per-row target value (classification label or regression value).
    target: Vec<f32>,
}

impl<T> Dataset<T> {
    /// Build a new dataset from its parts.
    pub fn new(features: Vec<Feature>, inputs: Vec<Vec<T>>, target: Vec<f32>) -> Result<Self> {
        if features.is_empty() {
            bail!("features are not set");
        }
        if features.len() != inputs.len() {
            bail!("input columns do not match features");
        }
        let rows = inputs[0].len();
        if inputs.iter().any(|col| col.len() != rows) {
            bail!("input matrix is not dense");
        }
        if !target.is_empty() && target.len() != rows {
            bail!("target size does not match inputs");
        }

        Ok(Self {
            features,
            inputs,
            target,
        })
    }

    /// Number of columns in the dataset.
    pub fn columns(&self) -> usize {
        self.inputs.len()
    }

    /// Number of rows in the dataset.
    pub fn rows(&self) -> usize {
        self.inputs.first().map(|x| x.len()).unwrap()
    }

    /// Get features reference.
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Get inputs reference.
    pub fn inputs(&self) -> &[Vec<T>] {
        &self.inputs
    }

    /// Get target data (if present).
    pub fn target(&self) -> &[f32] {
        &self.target
    }

    /// Get inputs as mutable slices.
    pub fn slices_mut(&mut self) -> Vec<&mut [T]> {
        self.inputs.iter_mut().map(|x| x.as_mut_slice()).collect()
    }

    /// Get target as mutable slice.
    pub fn target_mut(&mut self) -> &mut [f32] {
        &mut self.target
    }

    /// Get inputs as mutable reference.
    #[allow(unused)]
    pub(crate) fn inputs_mut(&mut self) -> &mut [Vec<T>] {
        &mut self.inputs
    }
}

impl<T> std::fmt::Debug for Dataset<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dataset with {} columns and {} rows",
            self.columns(),
            self.rows(),
        )
    }
}

/// A single row of the dataset with the per-iteration loss derivatives.
#[derive(Clone, Copy)]
pub struct RowIndex<I: IndexT, O: Objective> {
    /// Row position, dataset-wide or sampled-subset-relative.
    index: I,
    /// Stored gradient/hessian values.
    deriv: O,
}

impl<I: IndexT, O: Objective> RowIndex<I, O> {
    /// Constructor; `index` should be validated by the user.
    pub fn new(index: usize, deriv: O) -> Self {
        RowIndex {
            index: I::from_usize(index),
            deriv,
        }
    }

    /// Construct new index vector.
    pub fn new_array(size: usize) -> Vec<Self> {
        (0..size)
            .map(|idx| RowIndex::new(idx, O::derive(0.0, 0.0)))
            .collect()
    }

    /// Get index value.
    pub fn as_usize(&self) -> usize {
        self.index.to_usize()
    }

    /// Get gradient/hessian values.
    pub fn grad_hess(&self) -> (f32, f32) {
        self.deriv.grad_hess()
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
        let y_hat = sigmoid(y_pred);
        -(y * y_hat.ln() + (1.0 - y) * (1.0 - y_hat).ln())
    }

    /// Gradient: y^ - y; Hessian: y^ * (1 - y^)
    fn derive(y: f32, y_pred: f32) -> Self {
        let y_hat = sigmoid(y_pred);
        Self(y_hat - y, y_hat * (1.0 - y_hat))
    }

    fn grad_hess(&self) -> (f32, f32) {
        (self.0, self.1)
    }
}

/// The sigmoid of `y`: maps a raw score to a probability in (0, 1).
fn sigmoid(y: f32) -> f32 {
    1.0 / (1.0 + (-y).exp())
}

/// Build an empty dataset (used in tests).
#[allow(unused)]
pub(crate) fn create_test_dataset(columns: usize, rows: usize, bins: usize) -> Dataset<u16> {
    let bins = (0..bins).map(|i| i as f32).collect::<Vec<_>>();
    Dataset::<u16>::new(
        vec![Feature::linear("".into(), bins); columns],
        vec![vec![0; rows]; columns],
        vec![0.0; rows],
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_feature() {
        let feature = Feature::linear("name".into(), vec![1., 2.]);
        assert_eq!(feature.name(), "name");
        assert_eq!(feature.kind(), FeatureKind::Linear);
        assert_eq!(feature.bins(), &[1., 2.]);

        let feature = Feature::onehot("other".into(), vec![]);
        assert_eq!(feature.name(), "other");
        assert_eq!(feature.kind(), FeatureKind::OneHot);
        assert_eq!(feature.bins(), &[]);
    }

    #[test]
    fn create_dataset() {
        let features = vec![
            Feature::linear("a".into(), vec![]),
            Feature::onehot("b".into(), vec![]),
        ];
        let empty = Dataset::<u8>::new(features.clone(), vec![vec![]; 2], vec![]).unwrap();
        assert_eq!(empty.columns(), 2);
        assert_eq!(empty.rows(), 0);

        // Small dataset (`u8` bins).
        let input = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let data = Dataset::<u8>::new(features.clone(), input, vec![1.0; 3]).unwrap();
        assert_eq!(data.columns(), 2);
        assert_eq!(data.rows(), 3);

        assert_eq!(data.features()[0].name(), "a");
        assert_eq!(data.inputs()[0], &[0, 1, 2]);
        assert_eq!(data.target(), &[1.0, 1.0, 1.0]);

        // Large dataset (`u16` bins, target offloaded).
        let input = vec![vec![0; 1000]; 2];
        let mut data = Dataset::<u16>::new(features, input, vec![]).unwrap();
        assert_eq!(data.columns(), 2);
        assert_eq!(data.rows(), 1000);

        data.slices_mut()[0][0] = 6;
        assert_eq!(data.inputs()[0][0], 6);
        assert_eq!(data.target_mut(), &mut []);
        assert_eq!(format!("{data:?}"), "dataset with 2 columns and 1000 rows");
    }

    #[test]
    fn invalid_dataset() {
        let features = vec![Feature::default(); 2];
        let input = vec![vec![0; 3]; 2];
        let create = |f, x, y| Dataset::<u8>::new(f, x, y);

        let bad_features = create(vec![], vec![], vec![]);
        assert_eq!(
            bad_features.unwrap_err().to_string(),
            "features are not set"
        );

        let bad_columns = create(vec![Feature::default()], input.clone(), vec![]);
        assert_eq!(
            bad_columns.unwrap_err().to_string(),
            "input columns do not match features"
        );

        let bad_matrix = create(features.clone(), vec![vec![0; 3], vec![1]], vec![]);
        assert_eq!(
            bad_matrix.unwrap_err().to_string(),
            "input matrix is not dense"
        );

        let bad_target = create(features.clone(), input.clone(), vec![1.0]);
        assert_eq!(
            bad_target.unwrap_err().to_string(),
            "target size does not match inputs"
        );
    }

    #[test]
    fn create_rowindex() {
        let index = RowIndex::<u32, SquaredLoss>::new_array(10);
        for (idx, item) in index.iter().enumerate() {
            assert_eq!(item.as_usize(), idx);
            assert_eq!(item.grad_hess(), (0.0, 1.0));
        }

        let other = RowIndex::<u64, LogLoss>::new_array(1);
        assert_eq!(other[0].as_usize(), 0);
        assert_eq!(other[0].grad_hess(), (0.5, 0.25));
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
