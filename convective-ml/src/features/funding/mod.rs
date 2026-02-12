//! Funding rate feature.

use crate::features::{Feature, FeatureCategory, FeatureError, MarketConfig};
use atelier_data::funding::FundingRate;
use std::any::Any;

// ---------------------------------------------------------------------------
// FundingRateFeature
// ---------------------------------------------------------------------------

/// Raw signed funding rate, scaled to basis points (×10 000).
///
/// Positive ⇒ longs pay shorts, negative ⇒ shorts pay longs. Reflects
/// the cost asymmetry between long and short positioning.
#[derive(Debug, Clone)]
pub struct FundingRateFeature;

impl Feature for FundingRateFeature {
    type Input = FundingRate;
    type Output = f64;
    type Config = MarketConfig;

    fn name(&self) -> &'static str {
        "funding_rate"
    }

    fn description(&self) -> &'static str {
        "Signed funding rate in basis points (×10000)"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Flow
    }

    fn compute(
        &self,
        fr: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        // Scale to bps for better numerical range in downstream models
        Ok(fr.funding_rate * 10_000.0)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
