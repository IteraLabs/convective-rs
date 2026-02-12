//! convective-ml
//!
//! Distributed Machine Learning Modeling for the convective-rs Framework.

/// Loss function engineering
pub mod functions;

/// Convex Linear Models
pub mod models;

/// Optimizers and Learning Algorithms
pub mod optimizers;

/// Various metrics
pub mod metrics;

/// Features computation
pub mod features;

// Re-export the main functionality
pub use features::{
    Feature, FeatureCategory, FeatureError, FeatureSelector, FeaturesOutput,
    MarketConfig, OrderbookConfig, compute_features, compute_features_with_config,
    compute_single_orderbook,
};

// Re-export multi-source compute
pub use features::compute_market::{ALL_FEATURE_NAMES, compute_all_features};

// Re-export the registries
pub use features::registry::{
    LIQUIDATION_FEATURES, MARKET_FEATURES, ORDERBOOK_FEATURES, TRADE_FEATURES,
};

// Re-export model layer essentials
pub use models::{ComputeBackend, Model, ModelMode, NalgebraBackend};

#[cfg(any(feature = "torch", doc))]
#[cfg_attr(docsrs, doc(cfg(feature = "torch")))]
pub use models::TorchBackend;
