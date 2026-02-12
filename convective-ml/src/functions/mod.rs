/// Concrete loss functions.
pub mod cost;
/// Loss function trait.
pub mod interface;

pub use cost::*;
pub use interface::{LossFunction, LossOutput};
