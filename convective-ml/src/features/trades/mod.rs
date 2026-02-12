//! Trade-flow features.
//!
//! These features compute over a `Vec<Trade>` (all trades within one
//! synchronization period) and return a scalar `f64`.

use crate::features::{Feature, FeatureCategory, FeatureError, MarketConfig};
use atelier_data::{datasets, trades::Trade};
use std::any::Any;

// ---------------------------------------------------------------------------
// TradeIntensity
// ---------------------------------------------------------------------------

/// Total trade volume in the period.
///
/// High trade intensity signals urgency / market participation.
#[derive(Debug, Clone)]
pub struct TradeIntensityFeature;

impl Feature for TradeIntensityFeature {
    type Input = [Trade];
    type Output = f64;
    type Config = MarketConfig;

    fn name(&self) -> &'static str {
        "trade_intensity"
    }

    fn description(&self) -> &'static str {
        "Total trade volume (sum of amounts) in the period"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Flow
    }

    fn compute(
        &self,
        trades: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if trades.is_empty() {
            return Ok(0.0);
        }
        let total: f64 = trades.iter().map(|t| t.amount).sum();
        Ok(datasets::truncate_to_decimal(total, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// TradeDirectionImbalance
// ---------------------------------------------------------------------------

/// Net aggressor direction: (buy_volume - sell_volume) / total_volume.
///
/// Range [-1, 1].  Positive ⇒ net buying pressure, negative ⇒ net selling.
#[derive(Debug, Clone)]
pub struct TradeDirectionImbalanceFeature;

impl Feature for TradeDirectionImbalanceFeature {
    type Input = [Trade];
    type Output = f64;
    type Config = MarketConfig;

    fn name(&self) -> &'static str {
        "trade_direction_imbalance"
    }

    fn description(&self) -> &'static str {
        "Signed net aggressor imbalance: (buy_vol - sell_vol) / total_vol"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Flow
    }

    fn compute(
        &self,
        trades: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if trades.is_empty() {
            return Ok(0.0);
        }

        let mut buy_vol = 0.0_f64;
        let mut sell_vol = 0.0_f64;

        for t in trades {
            match t.side.as_str() {
                "Buy" => buy_vol += t.amount,
                "Sell" => sell_vol += t.amount,
                _ => {}
            }
        }

        let total = buy_vol + sell_vol;
        if total == 0.0 {
            return Ok(0.0);
        }

        let imb = (buy_vol - sell_vol) / total;
        Ok(datasets::truncate_to_decimal(imb, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
