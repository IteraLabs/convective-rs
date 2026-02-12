//! Linear (logistic-regression) model, generic over [`ComputeBackend`].

use super::backend::{ComputeBackend, NalgebraBackend, NalgebraError};
use super::interface::{Model, ModelMode};
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// LinearModel
// ---------------------------------------------------------------------------

/// A single-layer linear model: `z = X w + b`.
///
/// Weights and bias are stored as `B::Tensor`.  The struct is generic over
/// the compute backend so the same model definition works with `nalgebra`
/// (default) or `tch` (`--features torch`).
///
/// # Distributed extensibility
///
/// [`LinearModel`] exposes `weights` and `bias` publicly for future expansions
#[derive(Debug)]
pub struct LinearModel<B: ComputeBackend> {
    pub id: String,
    pub weights: B::Tensor,
    pub bias: B::Tensor,
    pub mode: ModelMode,
    _backend: PhantomData<B>,
}

impl<B: ComputeBackend> LinearModel<B> {
    /// Create a [`LinearModelBuilder`] for the given input dimensionality.
    pub fn builder(input_dim: usize) -> LinearModelBuilder<B> {
        LinearModelBuilder::new(input_dim)
    }
}

// ---------------------------------------------------------------------------
// Builder (backend-agnostic skeleton)
// ---------------------------------------------------------------------------

/// Builder for [`LinearModel`].
///
/// Call [`glorot_uniform_init`](LinearModelBuilder::glorot_uniform_init) to
/// materialise the model.  That method is implemented once per backend.
#[derive(Debug)]
pub struct LinearModelBuilder<B: ComputeBackend> {
    id: Option<String>,
    input_dim: usize,
    _backend: PhantomData<B>,
}

impl<B: ComputeBackend> LinearModelBuilder<B> {
    pub fn new(input_dim: usize) -> Self {
        Self {
            id: None,
            input_dim,
            _backend: PhantomData,
        }
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }
}

// =========================================================================
// Nalgebra implementation
// =========================================================================

impl LinearModelBuilder<NalgebraBackend> {
    /// Initialise weights with Glorot-uniform and bias to zero.
    pub fn glorot_uniform_init(self) -> LinearModel<NalgebraBackend> {
        use nalgebra::DMatrix;
        use rand::Rng;

        let m = self.input_dim;
        let limit = (6.0_f64).sqrt() / ((m + 1) as f64).sqrt();
        let mut rng = rand::rng();

        let weights = DMatrix::from_fn(m, 1, |_, _| rng.random_range(-limit..limit));
        let bias = DMatrix::zeros(1, 1);

        LinearModel {
            id: self.id.unwrap_or_default(),
            weights,
            bias,
            mode: ModelMode::Training,
            _backend: PhantomData,
        }
    }
}

impl Model<NalgebraBackend> for LinearModel<NalgebraBackend> {
    fn id(&self) -> &str {
        &self.id
    }

    fn mode(&self) -> ModelMode {
        self.mode
    }

    fn set_mode(&mut self, mode: ModelMode) {
        self.mode = mode;
    }

    #[tracing::instrument(skip(self, input), fields(model_id = %self.id, mode = ?self.mode))]
    fn forward(&self, input: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
        let z = input * &self.weights; // (n, m) × (m, 1) → (n, 1)
        let b = self.bias[(0, 0)];
        z.add_scalar(b)
    }

    #[tracing::instrument(skip(self), fields(model_id = %self.id))]
    fn save_model(&self, path: &str) -> Result<(), NalgebraError> {
        use serde_json::json;
        use std::io::Write;

        let w_data: Vec<f64> = self.weights.iter().copied().collect();
        let b_data: Vec<f64> = self.bias.iter().copied().collect();

        let payload = json!([
            {
                "name": "weights",
                "rows": self.weights.nrows(),
                "cols": self.weights.ncols(),
                "data": w_data
            },
            {
                "name": "bias",
                "rows": self.bias.nrows(),
                "cols": self.bias.ncols(),
                "data": b_data
            }
        ]);

        let mut file = std::fs::File::create(path)?;
        file.write_all(payload.to_string().as_bytes())?;
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(model_id = %self.id))]
    fn load_model(&mut self, path: &str) -> Result<(), NalgebraError> {
        let contents = std::fs::read_to_string(path)?;
        let entries: Vec<serde_json::Value> = serde_json::from_str(&contents)?;

        for entry in entries {
            let name = entry["name"]
                .as_str()
                .ok_or_else(|| NalgebraError::Shape("missing name".into()))?;
            let rows = entry["rows"]
                .as_u64()
                .ok_or_else(|| NalgebraError::Shape("missing rows".into()))?
                as usize;
            let cols = entry["cols"]
                .as_u64()
                .ok_or_else(|| NalgebraError::Shape("missing cols".into()))?
                as usize;
            let data: Vec<f64> = entry["data"]
                .as_array()
                .ok_or_else(|| NalgebraError::Shape("missing data".into()))?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect();

            let mat = nalgebra::DMatrix::from_column_slice(rows, cols, &data);
            match name {
                "weights" => self.weights = mat,
                "bias" => self.bias = mat,
                other => {
                    return Err(NalgebraError::Shape(format!(
                        "unexpected tensor name: {other}"
                    )));
                }
            }
        }
        Ok(())
    }
}

// =========================================================================
// Torch implementation
// =========================================================================

#[cfg(feature = "torch")]
use super::backend::TorchBackend;

#[cfg(feature = "torch")]
impl LinearModelBuilder<TorchBackend> {
    /// Initialise weights with Glorot-uniform and bias to zero.
    ///
    /// Weights and bias are created with `requires_grad = true`.
    pub fn glorot_uniform_init(self) -> LinearModel<TorchBackend> {
        let m = self.input_dim as i64;
        let limit = (6.0_f64).sqrt() / ((self.input_dim + 1) as f64).sqrt();

        let rand_weights =
            tch::Tensor::rand([m], (tch::Kind::Float, tch::Device::Cpu)) * limit;
        let weights = rand_weights.set_requires_grad(true);

        let bias = tch::Tensor::zeros([1], (tch::Kind::Float, tch::Device::Cpu))
            .set_requires_grad(true);

        LinearModel {
            id: self.id.unwrap_or_default(),
            weights,
            bias,
            mode: ModelMode::Training,
            _backend: PhantomData,
        }
    }
}

#[cfg(feature = "torch")]
impl LinearModel<TorchBackend> {
    /// Clone parameters without sharing gradient state.
    pub fn shallow_clone(&self) -> Self {
        LinearModel {
            id: self.id.clone(),
            weights: self.weights.shallow_clone(),
            bias: self.bias.shallow_clone(),
            mode: self.mode,
            _backend: PhantomData,
        }
    }

    /// Forward pass with externally-supplied parameters.
    ///
    /// Useful for distributed averaging where a coordinator holds
    /// aggregated weights.
    pub fn forward_with_params(
        &self,
        x: &tch::Tensor,
        weights: &tch::Tensor,
        bias: &tch::Tensor,
    ) -> tch::Tensor {
        x.matmul(weights) + bias
    }
}

#[cfg(feature = "torch")]
impl Model<TorchBackend> for LinearModel<TorchBackend> {
    fn id(&self) -> &str {
        &self.id
    }

    fn mode(&self) -> ModelMode {
        self.mode
    }

    fn set_mode(&mut self, mode: ModelMode) {
        self.mode = mode;
    }

    #[tracing::instrument(skip(self, input), fields(model_id = %self.id, mode = ?self.mode))]
    fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        if self.mode == ModelMode::Inference {
            tch::no_grad(|| {
                let z = input.matmul(&self.weights).to_kind(tch::Kind::Float)
                    + &self.bias.to_kind(tch::Kind::Float);
                z.to_kind(tch::Kind::Float)
            })
        } else {
            let z = input.matmul(&self.weights).to_kind(tch::Kind::Float)
                + &self.bias.to_kind(tch::Kind::Float);
            z.to_kind(tch::Kind::Float)
        }
    }

    #[tracing::instrument(skip(self), fields(model_id = %self.id))]
    fn save_model(&self, path: &str) -> Result<(), tch::TchError> {
        let state_dict = vec![
            ("weights".to_string(), self.weights.shallow_clone()),
            ("bias".to_string(), self.bias.shallow_clone()),
        ];
        tch::Tensor::save_multi(&state_dict, path)
    }

    #[tracing::instrument(skip(self), fields(model_id = %self.id))]
    fn load_model(&mut self, path: &str) -> Result<(), tch::TchError> {
        let state_dict = tch::Tensor::load_multi(path)?;
        for (name, tensor) in state_dict {
            match name.as_str() {
                "weights" => self.weights = tensor,
                "bias" => self.bias = tensor,
                _ => {
                    return Err(tch::TchError::FileFormat(format!(
                        "unexpected tensor: {name}"
                    )));
                }
            }
        }
        Ok(())
    }
}
