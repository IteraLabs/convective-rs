//! Compute backend abstraction for the model layer.
//!
//! Provides [`ComputeBackend`] trait with two implementations:
//! - [`NalgebraBackend`]: Default backend using `nalgebra::DMatrix<f64>`.
//! - [`TorchBackend`]: Optional backend using `tch::Tensor`, enabled via `--features torch`.

use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// Abstraction over linear-algebra / tensor backends.
///
/// The trait is intentionally minimal — construction helpers and shape
/// inspection only.  All numerics live in the per-backend `impl` blocks of
/// [`Model`](super::interface::Model),
/// [`LossFunction`](crate::functions::LossFunction), and
/// [`Optimizer`](crate::optimizers::Optimizer).
pub trait ComputeBackend: Sized + Send + Sync + 'static {
    /// The n-dimensional array / tensor type used by this backend.
    type Tensor: Debug + Send;

    /// Backend-specific error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Build a tensor from row-major `Vec<Vec<f64>>` (feature matrices).
    fn from_row_vecs(data: &[Vec<f64>]) -> Self::Tensor;

    /// Build a column-vector tensor from a flat slice (targets / labels).
    fn from_slice(data: &[f64]) -> Self::Tensor;

    /// Human-readable shape description (for tracing / debug).
    fn shape_info(t: &Self::Tensor) -> String;
}

// ---------------------------------------------------------------------------
// Nalgebra backend (always available)
// ---------------------------------------------------------------------------

/// Default compute backend backed by `nalgebra::DMatrix<f64>`.
#[derive(Debug, Clone, Copy)]
pub struct NalgebraBackend;

/// Error type for [`NalgebraBackend`] operations.
#[derive(Debug)]
pub enum NalgebraError {
    /// I/O error during save / load.
    Io(std::io::Error),
    /// JSON (de)serialisation error.
    Json(serde_json::Error),
    /// Tensor shape mismatch.
    Shape(String),
}

impl std::fmt::Display for NalgebraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
            Self::Shape(msg) => write!(f, "shape mismatch: {msg}"),
        }
    }
}

impl std::error::Error for NalgebraError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Shape(_) => None,
        }
    }
}

impl From<std::io::Error> for NalgebraError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for NalgebraError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl ComputeBackend for NalgebraBackend {
    type Tensor = nalgebra::DMatrix<f64>;
    type Error = NalgebraError;

    fn from_row_vecs(data: &[Vec<f64>]) -> Self::Tensor {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        let flat: Vec<f64> = data.iter().flat_map(|r| r.iter().copied()).collect();
        nalgebra::DMatrix::from_row_slice(rows, cols, &flat)
    }

    fn from_slice(data: &[f64]) -> Self::Tensor {
        nalgebra::DMatrix::from_column_slice(data.len(), 1, data)
    }

    fn shape_info(t: &Self::Tensor) -> String {
        format!("({}, {})", t.nrows(), t.ncols())
    }
}

// ---------------------------------------------------------------------------
// Torch backend (feature-gated)
// ---------------------------------------------------------------------------

/// Compute backend backed by `tch::Tensor` (libtorch / PyTorch C++ API).
///
/// Enabled with `--features torch`.
#[cfg(any(feature = "torch", doc))]
#[cfg_attr(docsrs, doc(cfg(feature = "torch")))]
#[derive(Debug, Clone, Copy)]
pub struct TorchBackend;

#[cfg(feature = "torch")]
impl ComputeBackend for TorchBackend {
    type Tensor = tch::Tensor;
    type Error = tch::TchError;

    fn from_row_vecs(data: &[Vec<f64>]) -> Self::Tensor {
        let rows = data.len() as i64;
        let cols = if rows > 0 { data[0].len() as i64 } else { 0 };
        let flat: Vec<f64> = data.iter().flat_map(|r| r.iter().copied()).collect();
        tch::Tensor::from_slice(&flat)
            .reshape([rows, cols])
            .to_kind(tch::Kind::Float)
    }

    fn from_slice(data: &[f64]) -> Self::Tensor {
        tch::Tensor::from_slice(data).to_kind(tch::Kind::Float)
    }

    fn shape_info(t: &Self::Tensor) -> String {
        format!("{:?}", t.size())
    }
}
