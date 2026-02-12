use crate::features::errors::FeatureError;
use serde::{Deserialize, Serialize};
use std::any::Any;

/// Core trait that all features must implement
pub trait Feature: Send + Sync + 'static {
    type Input: ?Sized;
    type Output: Clone + Send + 'static;
    type Config: Default + Clone + Send + Sync + 'static;

    /// Unique identifier for this feature
    fn name(&self) -> &'static str;

    /// Human-readable description
    fn description(&self) -> &'static str;

    /// Feature category for organization
    fn category(&self) -> FeatureCategory;

    /// Compute the feature value
    fn compute(
        &self,
        input: &Self::Input,
        config: &Self::Config,
    ) -> Result<Self::Output, FeatureError>;

    /// Default configuration for this feature
    fn default_config(&self) -> Self::Config {
        Self::Config::default()
    }

    /// Dependencies on other features (optional)
    fn dependencies(&self) -> Vec<&'static str> {
        vec![]
    }

    /// Type erasure for registry storage
    fn as_any(&self) -> &dyn Any;
}

#[derive(Debug, Eq, Hash, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureCategory {
    Spread,
    Price,
    Volume,
    Liquidity,
    Imbalance,
    Volatility,
    Flow,
    Timing,
}

/// Configuration for orderbook-based features
#[derive(Debug, Clone)]
pub struct OrderbookConfig {
    pub depth: usize,
    pub bps: f64,
}

impl Default for OrderbookConfig {
    fn default() -> Self {
        Self {
            depth: 5,
            bps: 0.001, // 10 bps
        }
    }
}

/// Configuration for multi-source / market-snapshot features.
///
/// Features that compute over `MarketSnapshot` use this config.
#[derive(Debug, Clone)]
pub struct MarketConfig {
    /// Orderbook depth for composite features that reference the book.
    pub depth: usize,
    /// Basis-point tolerance for price-band features.
    pub bps: f64,
}

impl Default for MarketConfig {
    fn default() -> Self {
        Self {
            depth: 5,
            bps: 0.001,
        }
    }
}
