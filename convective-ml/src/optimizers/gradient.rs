//! Gradient-descent family of optimisers, generic over [`ComputeBackend`].

use crate::models::backend::{ComputeBackend, NalgebraBackend};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Optimiser that applies a gradient update to model parameters.
///
/// Generic over `B` so that the parameter update logic can differ between
/// nalgebra (direct subtraction) and torch (`no_grad` context).
pub trait Optimizer<B: ComputeBackend>: std::fmt::Debug + Send {
    /// Apply one gradient-descent step.
    fn step(
        &self,
        weights: &mut B::Tensor,
        bias: &mut B::Tensor,
        weight_grad: &B::Tensor,
        bias_grad: &B::Tensor,
    );
}

// ---------------------------------------------------------------------------
// Gradient Descent
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GradientDescent {
    pub id: String,
    pub learning_rate: f64,
}

impl GradientDescent {
    pub fn builder() -> GradientDescentBuilder {
        GradientDescentBuilder::new()
    }
}

// --- Nalgebra impl ---

impl Optimizer<NalgebraBackend> for GradientDescent {
    fn step(
        &self,
        weights: &mut nalgebra::DMatrix<f64>,
        bias: &mut nalgebra::DMatrix<f64>,
        weight_grad: &nalgebra::DMatrix<f64>,
        bias_grad: &nalgebra::DMatrix<f64>,
    ) {
        *weights -= weight_grad * self.learning_rate;
        *bias -= bias_grad * self.learning_rate;
    }
}

// --- Torch impl ---

#[cfg(feature = "torch")]
use crate::models::backend::TorchBackend;

#[cfg(feature = "torch")]
impl Optimizer<TorchBackend> for GradientDescent {
    fn step(
        &self,
        weights: &mut tch::Tensor,
        bias: &mut tch::Tensor,
        weight_grad: &tch::Tensor,
        bias_grad: &tch::Tensor,
    ) {
        tch::no_grad(|| {
            let _ = weights.f_sub_(&(weight_grad * self.learning_rate));
            let _ = bias.f_sub_(&(bias_grad * self.learning_rate));
        });
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct GradientDescentBuilder {
    id: Option<String>,
    learning_rate: Option<f64>,
}

impl Default for GradientDescentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientDescentBuilder {
    pub fn new() -> Self {
        GradientDescentBuilder {
            id: None,
            learning_rate: None,
        }
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    pub fn build(self) -> Result<GradientDescent, &'static str> {
        let id = self.id.ok_or("Missing id")?;
        let learning_rate = self.learning_rate.ok_or("Missing learning_rate")?;
        Ok(GradientDescent { id, learning_rate })
    }
}
