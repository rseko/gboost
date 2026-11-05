//! Public booster API: an object-safe `Booster` trait, its two constructors,
//! and the concrete `BoostModel` implementation.
//!
//! The constructors erase the bin-index (`T`), row-index (`I`), and objective
//! (`O`) types of the underlying `BoostModel` into `Box<dyn Booster<F>>`, so
//! callers program against one interface regardless of which concrete booster
//! was built. `F` is the float type of the raw feature columns passed to
//! `predict`; it is the only type parameter that remains visible.
//!
//! The dispatch helpers pick the concrete types based on the options:
//! - `dispatch_bin_type` chooses the bin-index width (`u8` or `u16`).
//! - `dispatch_index_type` chooses the row-index width (`u32` by default;
//!   `u64` when the target exceeds `u32::MAX` rows and the `large-index`
//!   feature is enabled).
//! - `dispatch_objective` instantiates the objective.

use crate::boost_model::{BinT, BoostModel, BoostResult};
use crate::dataset::{Feature, IndexT, LogLoss, Objective, SquaredLoss};
use crate::options::{ObjectiveKind, Options};
use crate::quantize::QuantFloat;

/// Object-safe interface to a trained gradient-boosting model.
///
/// `F` is the float type of the raw feature columns passed to
/// [`Booster::predict`]. The trait is `Send`, so `Box<dyn Booster<F>>` can
/// be shared across threads.
pub trait Booster<F: QuantFloat>: Send {
    /// Grow up to `max_rounds` boosting rounds in a single call.
    fn fit(&mut self, max_rounds: usize) -> BoostResult<()>;

    /// Grow one boosting round and fold the new tree into the model.
    ///
    /// Returns `Ok(false)` when the round produced no tree (no gain), which
    /// callers may use to decide when to stop boosting.
    fn fit_next(&mut self) -> BoostResult<bool>;

    /// Score a batch of raw (unquantized) feature columns.
    ///
    /// `inputs` holds one column per feature, in feature order; every column
    /// must have the same length. Each row's prediction is the configured
    /// base score plus the sum of the leaf values the row lands in.
    fn predict(&self, inputs: &[&[F]]) -> BoostResult<Vec<f32>>;
}

/// Build a boxed, object-safe booster over in-memory training columns.
///
/// Feature names and kinds are synthesized (`linear` features named by
/// position). Use [`new_with_features`] to supply explicit descriptors.
///
/// # Errors
///
/// Returns an error if the data is inconsistent (see [`BoostError::Dataset`])
/// or if the target is too large (see [`BoostError::DatasetTooLarge`]).
pub fn new_booster<F: QuantFloat>(
    options: Options,
    inputs: &[&[F]],
    target: &[f32],
) -> BoostResult<Box<dyn Booster<F>>> {
    dispatch_bin_type(options, inputs, target, None)
}

/// Build a boxed, object-safe booster over in-memory training columns with
/// explicit feature descriptors.
///
/// `features` must have one entry per input column. Only the name and kind
/// are carried over; the bin boundaries are always refit from the inputs.
///
/// # Errors
///
/// See [`new_booster`].
pub fn new_with_features<F: QuantFloat>(
    options: Options,
    inputs: &[&[F]],
    target: &[f32],
    features: Vec<Feature>,
) -> BoostResult<Box<dyn Booster<F>>> {
    dispatch_bin_type(options, inputs, target, Some(features))
}

/// Forward `fit_next` and `predict` from the trait to the concrete
/// `BoostModel`. The bounds here mirror those required by the concrete
/// methods; `F` is unconstrained at this layer beyond what the trait
/// already demands.
impl<F: QuantFloat, T: BinT, I: IndexT, O: Objective> Booster<F> for BoostModel<T, I, O> {
    fn fit(&mut self, max_rounds: usize) -> BoostResult<()> {
        for _ in 0..max_rounds {
            if !self.fit_next()? {
                break;
            }
        }
        Ok(())
    }

    fn fit_next(&mut self) -> BoostResult<bool> {
        BoostModel::<T, I, O>::fit_next(self)
    }

    fn predict(&self, inputs: &[&[F]]) -> BoostResult<Vec<f32>> {
        BoostModel::<T, I, O>::predict(self, inputs)
    }
}

/// Choose the bin-index width from the configured `max_bins`.
///
/// `u8` is sufficient when `max_bins <= 255`; otherwise `u16` is required
/// to hold bin indices up to `max_bins`.
fn dispatch_bin_type<F: QuantFloat>(
    options: Options,
    inputs: &[&[F]],
    target: &[f32],
    features: Option<Vec<Feature>>,
) -> BoostResult<Box<dyn Booster<F>>> {
    if options.boost_options.max_bins <= 255 {
        dispatch_index_type::<F, u8>(options, inputs, target, features, None)
    } else {
        dispatch_index_type::<F, u16>(options, inputs, target, features, None)
    }
}

/// Choose the row-index width from the target size.
///
/// `u32` is the default. When `target.len()` exceeds the `size_limit`
/// (default `u32::MAX`), a `u64` index is required, which is only available
/// under the `large-index` feature; otherwise `DatasetTooLarge` is returned.
fn dispatch_index_type<F, T>(
    options: Options,
    inputs: &[&[F]],
    target: &[f32],
    features: Option<Vec<Feature>>,
    size_limit: Option<usize>,
) -> BoostResult<Box<dyn Booster<F>>>
where
    F: QuantFloat,
    T: BinT + 'static,
{
    let size_limit = size_limit.unwrap_or(u32::MAX as usize);
    if target.len() <= size_limit {
        dispatch_objective::<F, T, u32>(options, inputs, target, features)
    } else {
        #[cfg(feature = "large-index")]
        {
            dispatch_objective::<F, T, u64>(options, inputs, target, features)
        }
        #[cfg(not(feature = "large-index"))]
        {
            Err(crate::boost_model::BoostError::DatasetTooLarge(size_limit))
        }
    }
}

/// Instantiate the concrete `BoostModel` with the configured objective and
/// box it behind the `Booster` trait object.
fn dispatch_objective<F, T, I>(
    options: Options,
    inputs: &[&[F]],
    target: &[f32],
    features: Option<Vec<Feature>>,
) -> BoostResult<Box<dyn Booster<F>>>
where
    F: QuantFloat,
    T: BinT + 'static,
    I: IndexT + 'static,
{
    Ok(match options.boost_options.objective {
        ObjectiveKind::LogLoss => Box::new(BoostModel::<T, I, LogLoss>::init_small(
            options, inputs, target, features,
        )?),
        ObjectiveKind::SquaredLoss => Box::new(BoostModel::<T, I, SquaredLoss>::init_small(
            options, inputs, target, features,
        )?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boost_model::FILENAME_MODEL;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    const NUM_ROWS: usize = 1_000;
    const NUM_COLS: usize = 4;

    fn build_data() -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut rng = StdRng::seed_from_u64(0);
        let data = (0..NUM_COLS)
            .map(|_| (0..NUM_ROWS).map(|_| rng.random_range(0.0..1.0)).collect())
            .collect();
        let target = (0..NUM_ROWS).map(|_| rng.random_range(0.0..1.0)).collect();
        (data, target)
    }

    #[test]
    fn new_booster_squared_loss() {
        let (data, target) = build_data();
        let inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

        let mut booster = new_booster::<f32>(
            Options::default().set_objective(ObjectiveKind::SquaredLoss),
            &inputs,
            &target,
        )
        .unwrap();
        assert_eq!(booster.fit_next().ok(), Some(true));

        let preds = booster.predict(&inputs).unwrap();
        assert_eq!(preds.len(), NUM_ROWS);
    }

    #[test]
    fn new_booster_log_loss() {
        let (data, target) = build_data();
        let inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

        let mut booster = new_booster::<f32>(
            Options::default().set_objective(ObjectiveKind::LogLoss),
            &inputs,
            &target,
        )
        .unwrap();
        assert_eq!(booster.fit_next().ok(), Some(true));

        let preds = booster.predict(&inputs).unwrap();
        assert_eq!(preds.len(), NUM_ROWS);
    }

    #[test]
    fn new_booster_u16_bins() {
        let (data, target) = build_data();
        let inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

        let mut booster =
            new_booster::<f32>(Options::default().set_max_bins(65535), &inputs, &target).unwrap();
        assert_eq!(booster.fit_next().ok(), Some(true));

        let preds = booster.predict(&inputs).unwrap();
        assert_eq!(preds.len(), NUM_ROWS);
    }

    #[test]
    fn new_with_features_test() {
        let (data, target) = build_data();
        let inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

        let features = (0..NUM_COLS)
            .map(|i| Feature::linear(Some(format!("f{i}")), vec![]))
            .collect();
        let mut booster =
            new_with_features::<f32>(Options::default(), &inputs, &target, features).unwrap();
        assert_eq!(booster.fit_next().ok(), Some(true));

        let preds = booster.predict(&inputs).unwrap();
        assert_eq!(preds.len(), NUM_ROWS);
    }

    #[test]
    fn large_index_size() {
        let result =
            dispatch_index_type::<f32, u8>(Options::default(), &[&[1.0]], &[1.0], None, Some(0));
        #[cfg(feature = "large-index")]
        assert!(result.is_ok());
        #[cfg(not(feature = "large-index"))]
        assert!(result.is_err());
    }

    #[test]
    fn error_propagation() {
        let squared_loss = Options::default().set_objective(ObjectiveKind::SquaredLoss);
        let res1 = dispatch_objective::<f32, u8, u32>(squared_loss, &[], &[], None);
        assert!(res1.is_err());

        let log_loss = Options::default().set_objective(ObjectiveKind::LogLoss);
        let res2 = dispatch_objective::<f32, u8, u32>(log_loss, &[], &[], None);
        assert!(res2.is_err());
    }

    #[test]
    fn fit_then_predict() {
        let (data, target) = build_data();
        let inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

        let mut booster = new_booster::<f32>(Options::default(), &inputs, &target).unwrap();
        booster.fit(50).unwrap();

        let preds = booster.predict(&inputs).unwrap();
        let mean_error = preds
            .iter()
            .zip(&target)
            .map(|(p, t)| (p - t).abs())
            .sum::<f32>()
            / NUM_ROWS as f32;
        // Deterministic seed and fixed rounds make the value stable.
        assert_eq!(mean_error, 0.0946346);
    }

    #[test]
    fn fit_error() {
        let (data, target) = build_data();
        let inputs = data.iter().map(|x| x.as_slice()).collect::<Vec<_>>();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let options = Options::default()
            .set_persist_path(Some(temp_dir.path().into()));

        let mut booster = new_booster::<f32>(options, &inputs, &target).unwrap();
        booster.fit(1).unwrap();
        std::fs::remove_file(temp_dir.path().join(FILENAME_MODEL)).unwrap();
        assert!(booster.fit(1).is_err());
    }
}
