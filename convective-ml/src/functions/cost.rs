//! Cross-entropy loss with per-backend gradient computation.

use super::interface::{LossFunction, LossOutput};
use crate::models::backend::NalgebraBackend;

// ---------------------------------------------------------------------------
// Regularisation helpers (kept for future use, backend-agnostic)
// ---------------------------------------------------------------------------

pub trait Regularized {
    fn id(&mut self, id: String);
    fn regularize(&self, weights: &[f64], operation: &RegType, params: &[f64]) -> f64;
}

#[derive(Debug, Copy, Clone)]
pub enum RegType {
    L1,
    L2,
    Elasticnet,
}

// ---------------------------------------------------------------------------
// CrossEntropy struct
// ---------------------------------------------------------------------------

/// Binary cross-entropy loss (with logits).
///
/// The struct itself is backend-agnostic.  Gradient computation is provided
/// by per-backend `impl LossFunction<B>` blocks below.
#[derive(Debug)]
pub struct CrossEntropy {
    pub id: String,
}

impl CrossEntropy {
    pub fn builder<'a>() -> CrossEntropyBuilder<'a> {
        CrossEntropyBuilder::new()
    }
}

impl Regularized for CrossEntropy {
    fn id(&mut self, id: String) {
        self.id = id;
    }

    fn regularize(&self, weights: &[f64], operation: &RegType, params: &[f64]) -> f64 {
        let r_c = params[0];
        let r_lambda = params[1];
        let l1: f64 = weights.iter().map(|w| w.abs()).sum();
        let l2: f64 = weights.iter().map(|w| w * w).sum();

        match operation {
            RegType::L1 => r_c * r_lambda * l1,
            RegType::L2 => r_c * r_lambda * l2,
            RegType::Elasticnet => r_c * (r_lambda * l1 + (1.0 - r_lambda) * l2),
        }
    }
}

// =========================================================================
// Nalgebra implementation
// =========================================================================

impl LossFunction<NalgebraBackend> for CrossEntropy {
    fn loss_and_gradients(
        &self,
        features: &nalgebra::DMatrix<f64>,
        logits: &nalgebra::DMatrix<f64>,
        targets: &nalgebra::DMatrix<f64>,
        _weights: &mut nalgebra::DMatrix<f64>,
        _bias: &mut nalgebra::DMatrix<f64>,
    ) -> LossOutput<NalgebraBackend> {
        let n = features.nrows() as f64;

        // --- Loss: numerically-stable BCE with logits ---
        // loss_i = max(x, 0) - x * y + ln(1 + exp(-|x|))
        let loss_value: f64 = logits
            .iter()
            .zip(targets.iter())
            .map(|(&x, &y)| x.max(0.0) - x * y + (1.0 + (-x.abs()).exp()).ln())
            .sum::<f64>()
            / n;

        // --- Gradients: closed-form for logistic regression ---
        // sigmoid(logits)
        let y_hat = logits.map(|x| 1.0 / (1.0 + (-x).exp()));
        // delta = y_hat - targets   (n, 1)
        let delta = &y_hat - targets;
        // dw = Xᵀ δ / n             (m, 1)
        let weight_grad = features.transpose() * &delta / n;
        // db = mean(δ)              (1, 1)
        let bias_grad =
            nalgebra::DMatrix::from_element(1, 1, delta.iter().sum::<f64>() / n);

        LossOutput {
            loss_value,
            weight_grad,
            bias_grad,
        }
    }
}

// =========================================================================
// Torch implementation
// =========================================================================

#[cfg(feature = "torch")]
use crate::models::backend::TorchBackend;

#[cfg(feature = "torch")]
impl LossFunction<TorchBackend> for CrossEntropy {
    fn loss_and_gradients(
        &self,
        _features: &tch::Tensor,
        logits: &tch::Tensor,
        targets: &tch::Tensor,
        weights: &mut tch::Tensor,
        bias: &mut tch::Tensor,
    ) -> LossOutput<TorchBackend> {
        // Clear accumulated gradients from previous iteration
        // (must happen BEFORE backward, not after — .grad()
        // returns a handle to the same storage that zero_grad
        // would wipe).
        weights.zero_grad();
        bias.zero_grad();

        // Compute BCE with logits (numerically stable,
        // autograd-tracked).
        // NOTE: do NOT call .set_requires_grad(true) on the
        // loss — that would promote it to a leaf tensor and
        // sever the autograd chain back to weights / bias.
        let loss = logits.binary_cross_entropy_with_logits::<&tch::Tensor>(
            targets,
            None,
            None,
            tch::Reduction::Mean,
        );

        let loss_value = loss.double_value(&[]);

        // Backward pass — populates .grad() on weights and bias
        loss.backward();

        let weight_grad = weights.grad();
        let bias_grad = bias.grad();

        LossOutput {
            loss_value,
            weight_grad,
            bias_grad,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CrossEntropyBuilder<'a> {
    id: Option<&'a str>,
}

impl<'a> Default for CrossEntropyBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> CrossEntropyBuilder<'a> {
    pub fn new() -> Self {
        CrossEntropyBuilder { id: None }
    }

    pub fn id(mut self, id: &'a str) -> Self {
        self.id = Some(id);
        self
    }

    pub fn build(self) -> Result<CrossEntropy, &'static str> {
        let id = self.id.ok_or("Missing id value")?;
        Ok(CrossEntropy { id: id.to_string() })
    }
}
