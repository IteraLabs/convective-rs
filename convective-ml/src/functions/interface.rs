//! Loss function trait, generic over [`crate::models::backend::ComputeBackend`].

use crate::models::backend::ComputeBackend;

/// Output of [`LossFunction::loss_and_gradients`].
#[derive(Debug)]
pub struct LossOutput<B: ComputeBackend> {
    /// Scalar loss value for logging / early-stopping.
    pub loss_value: f64,
    /// Gradient of the loss w.r.t. model weights.
    pub weight_grad: B::Tensor,
    /// Gradient of the loss w.r.t. model bias.
    pub bias_grad: B::Tensor,
}

/// A loss function that can compute both the loss *and* parameter gradients
/// in a single call.
///
/// The reason loss and gradients are fused:
///
/// * **Torch backend** — the gradient comes from autograd (`loss.backward()`),
///   which needs the loss tensor that was connected to the computation graph
///   through `weights` / `bias`.  Splitting loss and gradient computation
///   would require recomputing the forward pass.
///
/// * **Nalgebra backend** — we compute the closed-form gradient of BCE
///   directly from the logits, features, and targets.  No autograd exists.
///
/// Fusing both operations into one method keeps the training loop fully
/// generic while allowing each backend to use its natural strategy.
pub trait LossFunction<B: ComputeBackend>: std::fmt::Debug + Send {
    /// Compute loss value and parameter gradients.
    ///
    /// # Parameters
    ///
    /// * `features`  — input feature matrix  *(n × m)*
    /// * `logits`    — raw model output       *(n × 1)* or *(n,)*
    /// * `targets`   — ground-truth labels    *(n × 1)* or *(n,)*
    /// * `weights`   — model weight tensor    *(m × 1)* or *(m,)* — mutable
    ///   so the torch backend can read/clear `.grad()`.
    /// * `bias`      — model bias tensor      *(1 × 1)* or *(1,)* — same
    ///   reason.
    fn loss_and_gradients(
        &self,
        features: &B::Tensor,
        logits: &B::Tensor,
        targets: &B::Tensor,
        weights: &mut B::Tensor,
        bias: &mut B::Tensor,
    ) -> LossOutput<B>;
}
