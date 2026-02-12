//! CEX liquidation features.
//!
//! Liquidation events are forced position closures triggered by insufficient
//! margin. These features measure the magnitude and directional skew of
//! liquidation activity within a synchronization period.

use crate::features::{Feature, FeatureCategory, FeatureError, MarketConfig};
use atelier_data::{datasets, liquidations::Liquidation};
use std::any::Any;

// ---------------------------------------------------------------------------
// LiquidationPressure
// ---------------------------------------------------------------------------

/// Total liquidation notional (price × amount) in the period.
///
/// High pressure signals cascading forced exits, which tend to amplify
/// price moves.
#[derive(Debug, Clone)]
pub struct LiquidationPressureFeature;

impl Feature for LiquidationPressureFeature {
    type Input = [Liquidation];
    type Output = f64;
    type Config = MarketConfig;

    fn name(&self) -> &'static str {
        "liquidation_pressure"
    }

    fn description(&self) -> &'static str {
        "Total liquidation notional (price * amount) in the period"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Flow
    }

    fn compute(
        &self,
        liqs: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if liqs.is_empty() {
            return Ok(0.0);
        }

        let notional: f64 = liqs.iter().map(|l| l.price * l.amount).sum();
        Ok(datasets::truncate_to_decimal(notional, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// LiquidationImbalance
// ---------------------------------------------------------------------------

/// Directional imbalance of liquidations:
/// (buy_liq_volume - sell_liq_volume) / total_liq_volume.
///
/// Positive ⇒ more short liquidations ("Buy" side forced), negative ⇒
/// more long liquidations ("Sell" side forced).
#[derive(Debug, Clone)]
pub struct LiquidationImbalanceFeature;

impl Feature for LiquidationImbalanceFeature {
    type Input = [Liquidation];
    type Output = f64;
    type Config = MarketConfig;

    fn name(&self) -> &'static str {
        "liquidation_imbalance"
    }

    fn description(&self) -> &'static str {
        "Liquidation direction imbalance: (buy - sell) / total"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Imbalance
    }

    fn compute(
        &self,
        liqs: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if liqs.is_empty() {
            return Ok(0.0);
        }

        let mut buy_vol = 0.0_f64;
        let mut sell_vol = 0.0_f64;

        for l in liqs {
            match l.side.as_str() {
                "Buy" => buy_vol += l.amount,
                "Sell" => sell_vol += l.amount,
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
