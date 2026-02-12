/// Compute backend abstraction ([`NalgebraBackend`](crate::models::backend::NalgebraBackend),
/// [`TorchBackend`](crate::models::backend::TorchBackend)).
///
/// - [`NalgebraBackend`]: Default backend using `nalgebra` matrices.
/// - [`TorchBackend`]: Optional backend using `tch::Tensor`, enabled via `--features torch`.
pub mod backend;

/// Model trait and supporting types.
pub mod interface;

/// Linear (logistic-regression) model.
pub mod linear;

// Re-exports for convenience
pub use backend::{ComputeBackend, NalgebraBackend};
pub use interface::{Model, ModelMode};
pub use linear::{LinearModel, LinearModelBuilder};

#[cfg(any(feature = "torch", doc))]
#[cfg_attr(docsrs, doc(cfg(feature = "torch")))]
pub use backend::TorchBackend;
