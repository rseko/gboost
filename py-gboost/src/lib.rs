//! Python bindings for the `gboost` gradient boosting library.
//!
//! Exposes a `GradientBooster` class with a `fit(x, y)` / `fit_next` /
//! `predict(x)` API:
//! - `x` is a 2-D float32 matrix (`(rows, cols)`),
//! - `y` a 1-D float32 target vector,
//! - `predict` returns a 1-D float32 score array.
//!
//! The core works on column-major `inputs[col][row]` slices, so `x` should be
//! a Fortran-order array (e.g. `np.asfortranarray(x)`) for zero-copy; row-major
//! or strided inputs are transposed into column-major internally.

use gboost::boost_api::{Booster, new_booster, new_with_features};
use gboost::boost_model::BoostError;
use gboost::dataset::Feature;
use gboost::options::{ObjectiveKind, Options};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::collections::HashSet;
use std::path::PathBuf;

/// Gradient boosting learner backed by the Rust core.
#[pyclass]
struct GradientBooster {
    options: Options,
    iterations: usize,
    onehot_columns: Vec<usize>,
    booster: Option<Box<dyn Booster<f32>>>,
}

#[pymethods]
impl GradientBooster {
    #[new]
    #[pyo3(signature = (
        *,
        iterations = 100,
        objective = "squaredloss",
        base_score = 0.0,
        max_bins = 255,
        persist_path = None,
        parallel = true,
        seed = 0,
        learning_rate = 0.3,
        subsample = 1.0,
        colsample_bytree = 1.0,
        colsample_bylevel = 1.0,
        colsample_bynode = 1.0,
        max_depth = 6,
        max_leaves = 0,
        grow_policy = "depthwise",
        reg_alpha = 0.0,
        reg_lambda = 1.0,
        min_split_gain = 0.0,
        min_child_weight = 1.0,
        allow_missing_splits = false,
        onehot_columns = Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        iterations: usize,
        // BoosterOptions
        objective: &str,
        base_score: f32,
        max_bins: u16,
        persist_path: Option<&str>,
        parallel: bool,
        seed: u64,
        // BoostTreeOptions
        learning_rate: f64,
        subsample: f64,
        colsample_bytree: f64,
        colsample_bylevel: f64,
        colsample_bynode: f64,
        max_depth: usize,
        max_leaves: usize,
        grow_policy: &str,
        // BoostNodeOptions
        reg_alpha: f64,
        reg_lambda: f64,
        min_split_gain: f64,
        min_child_weight: f64,
        allow_missing_splits: bool,
        // Miscellaneous
        onehot_columns: Vec<usize>,
    ) -> PyResult<Self> {
        let objective = match objective.trim().to_ascii_lowercase().as_str() {
            "squaredloss" => ObjectiveKind::SquaredLoss,
            "logloss" => ObjectiveKind::LogLoss,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid objective {other:?}; expected 'squaredloss' or 'logloss'"
                )));
            }
        };

        let loss_guide = match grow_policy.trim().to_ascii_lowercase().as_str() {
            "depthwise" => false,
            "lossguide" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid grow_policy {other:?}; expected 'depthwise' or 'lossguide'"
                )));
            }
        };

        let options = Options::default()
            .set_objective(objective)
            .set_base_score(base_score)
            .set_max_bins(max_bins)
            .set_persist_path(persist_path.map(PathBuf::from))
            .set_parallel(parallel)
            .set_seed(seed)
            .set_learning_rate(learning_rate)
            .set_subsample(subsample)
            .set_colsample_bytree(colsample_bytree)
            .set_colsample_bylevel(colsample_bylevel)
            .set_colsample_bynode(colsample_bynode)
            .set_max_depth(max_depth)
            .set_max_leaves(max_leaves)
            .set_loss_guide(loss_guide)
            .set_reg_alpha(reg_alpha)
            .set_reg_lambda(reg_lambda)
            .set_min_split_gain(min_split_gain)
            .set_min_child_weight(min_child_weight)
            .set_allow_missing_splits(allow_missing_splits);

        Ok(Self {
            options,
            iterations,
            onehot_columns,
            booster: None,
        })
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let params = self.get_params(py)?;
        let pairs: Vec<String> = params
            .iter()
            .map(|(key, value)| {
                let key = key.str()?.to_string_lossy().into_owned();
                let value = value.repr()?.to_string_lossy().into_owned();
                Ok(format!("{key}={value}"))
            })
            .collect::<PyResult<_>>()?;
        Ok(format!("GradientBooster({})", pairs.join(", ")))
    }

    fn __str__(&self, py: Python) -> PyResult<String> {
        self.__repr__(py)
    }

    /// Return the constructor kwargs as a `dict` of `name -> value`, matching
    /// the `GradientBooster(...)` signature.
    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let o = &self.options;

        dict.set_item("iterations", self.iterations)?;
        dict.set_item(
            "objective",
            match o.boost_options.objective {
                ObjectiveKind::SquaredLoss => "squaredloss",
                ObjectiveKind::LogLoss => "logloss",
            },
        )?;
        dict.set_item("base_score", o.boost_options.base_score)?;
        dict.set_item("max_bins", o.boost_options.max_bins)?;
        dict.set_item(
            "persist_path",
            o.boost_options
                .persist_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
        )?;
        dict.set_item("parallel", o.boost_options.parallel)?;
        dict.set_item("seed", o.boost_options.seed)?;
        dict.set_item("learning_rate", o.tree_options.learning_rate)?;
        dict.set_item("subsample", o.tree_options.subsample)?;
        dict.set_item("colsample_bytree", o.tree_options.colsample_bytree)?;
        dict.set_item("colsample_bylevel", o.tree_options.colsample_bylevel)?;
        dict.set_item("colsample_bynode", o.tree_options.colsample_bynode)?;
        dict.set_item("max_depth", o.tree_options.max_depth)?;
        dict.set_item("max_leaves", o.tree_options.max_leaves)?;
        dict.set_item(
            "grow_policy",
            if o.tree_options.loss_guide {
                "lossguide"
            } else {
                "depthwise"
            },
        )?;
        dict.set_item("reg_alpha", o.node_options.reg_alpha)?;
        dict.set_item("reg_lambda", o.node_options.reg_lambda)?;
        dict.set_item("min_split_gain", o.node_options.min_split_gain)?;
        dict.set_item("min_child_weight", o.node_options.min_child_weight)?;
        dict.set_item("allow_missing_splits", o.node_options.allow_missing_splits)?;
        dict.set_item("onehot_columns", self.onehot_columns.clone())?;

        Ok(dict)
    }

    /// Run gradient boosting over `x` (rows x cols) toward `y`.
    ///
    /// Replaces any previously trained model; use `fit_next` to extend an
    /// existing one.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f32>,
        y: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let flat = column_major(&x);
        let rows = x.as_array().shape()[0];
        let inputs: Vec<&[f32]> = flat.chunks_exact(rows).collect();

        let target = target_flat(&y);
        if target.iter().any(|v| v.is_nan()) {
            return Err(PyValueError::new_err("target contains NaN values"));
        }

        let options = self.options.clone();
        let onehot: HashSet<usize> = self.onehot_columns.iter().copied().collect();
        let iterations = self.iterations;

        let booster = py
            .detach(|| -> Result<Box<dyn Booster<f32>>, PyErr> {
                let result = if onehot.is_empty() {
                    new_booster(options, &inputs, &target)
                } else {
                    let new_feature = |i: usize| match onehot.contains(&i) {
                        true => Feature::onehot("", None),
                        false => Feature::linear("", None),
                    };
                    let features = (0..inputs.len()).map(new_feature).collect();
                    new_with_features(options, &inputs, &target, features)
                };
                let mut b = result.map_err(boost_error_to_py)?;
                b.fit(iterations).map_err(boost_error_to_py)?;
                Ok(b)
            })?;

        self.booster = Some(booster);
        Ok(())
    }

    /// Grow one additional boosting round on the trained model.
    ///
    /// Returns `false` when the round produced no tree (no gain); use it to
    /// decide when to stop boosting.
    fn fit_next(&mut self, py: Python<'_>) -> PyResult<bool> {
        let booster = self
            .booster
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("model is not trained"))?;

        py.detach(|| booster.fit_next().map_err(boost_error_to_py))
    }

    /// Score `x` (rows x cols) with the trained model.
    ///
    /// Errors if the model is untrained or if `x` has a different number of
    /// columns than the training data. `x` may be row-major; a non-contiguous
    /// input is transposed internally.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let booster = self
            .booster
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not trained"))?;

        let flat = column_major(&x);
        let rows = x.as_array().shape()[0];
        let inputs: Vec<&[f32]> = flat.chunks_exact(rows).collect();

        let scores = py
            .detach(|| booster.predict(&inputs).map_err(boost_error_to_py))?;
        Ok(scores.into_pyarray(py).into_any())
    }

    /// Mean loss of the current predictions against the training labels.
    ///
    /// Returns `None` when the model has not been trained.
    fn loss(&self) -> Option<f32> {
        self.booster
            .as_ref()
            .map(|b| b.loss())
            .map(|a| a.iter().sum::<f32>() / a.len() as f32)
    }
}

/// Map a booster error to a Python exception: `OSError` for I/O failures,
/// `ValueError` otherwise.
fn boost_error_to_py(err: BoostError) -> PyErr {
    match err {
        BoostError::Io(_) => PyIOError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// A flat `f32` buffer: either a zero-copy slice or a copied `Vec`.
enum FlatF32<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

impl std::ops::Deref for FlatF32<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(v) => v,
        }
    }
}

/// Return `x` as a flat column-major slice. Zero-copy when `x` is already
/// Fortran-contiguous; otherwise transposes into a new `Vec`.
fn column_major<'a>(x: &'a PyReadonlyArray2<f32>) -> FlatF32<'a> {
    if x.is_fortran_contiguous() {
        return FlatF32::Borrowed(x.as_slice().unwrap());
    }
    let flat = x.as_array().t().iter().copied().collect::<Vec<f32>>();
    FlatF32::Owned(flat)
}

/// Return `y` as a flat slice. Zero-copy when contiguous; otherwise copies.
fn target_flat<'a>(y: &'a PyReadonlyArray1<f32>) -> FlatF32<'a> {
    if y.is_contiguous() {
        return FlatF32::Borrowed(y.as_slice().unwrap());
    }
    FlatF32::Owned(y.as_array().iter().copied().collect())
}

#[pymodule]
fn gboostrs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GradientBooster>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
