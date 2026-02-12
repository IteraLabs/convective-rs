use thiserror::Error;

#[derive(Error, Debug)]
pub enum FeatureError {
    #[error("Empty orderbook")]
    EmptyOrderbook,

    #[error("No trades in period")]
    NoTrades,

    #[error("No liquidations in period")]
    NoLiquidations,

    #[error("Insufficient depth: requested {requested}, available {available}")]
    InsufficientDepth { requested: usize, available: usize },

    #[error("Zero volume")]
    ZeroVolume,

    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    #[error("Computation error: {message}")]
    ComputationError { message: String },

    #[error("Feature not found: {name}")]
    FeatureNotFound { name: String },
}
