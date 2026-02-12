//! Model trait and supporting types.

use super::backend::ComputeBackend;

/// Whether a model is in training or inference mode.
///
/// In **Training** mode backends that support automatic differentiation
/// (e.g. `TorchBackend`) will track gradients through the forward pass.
///
/// In **Inference** mode gradient tracking is disabled, which reduces
/// memory usage and speeds up the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelMode {
    Training,
    Inference,
}

/// Core model abstraction, generic over the compute backend `B`.
///
/// Every concrete model (e.g. [`LinearModel`](super::linear::LinearModel))
/// implements this trait once per supported backend.
///
/// The trait is deliberately *object-safe* so that a future distributed
/// trainer can hold `Vec<Box<dyn Model<B>>>` (a swarm of heterogeneous
/// models).
///
/// `Sync` is intentionally **not** required because `tch::Tensor`
/// wraps a raw pointer that is `!Sync`.  Single-threaded training
/// works fine; distributed trainers that need cross-thread sharing
/// can wrap models in `Arc<Mutex<_>>`.
pub trait Model<B: ComputeBackend>: std::fmt::Debug + Send {
    /// Unique identifier for this model instance.
    fn id(&self) -> &str;

    /// Current execution mode.
    fn mode(&self) -> ModelMode;

    /// Switch between [`ModelMode::Training`] and [`ModelMode::Inference`].
    fn set_mode(&mut self, mode: ModelMode);

    /// Compute the forward pass.
    ///
    /// For a linear model this returns raw logits (no activation).
    fn forward(&self, input: &B::Tensor) -> B::Tensor;

    /// Persist model parameters to `path`.
    fn save_model(&self, path: &str) -> Result<(), B::Error>;

    /// Restore model parameters from `path`.
    fn load_model(&mut self, path: &str) -> Result<(), B::Error>;
}
