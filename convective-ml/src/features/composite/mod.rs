//! Composite features that combine orderbook and trade data.
//!
//! These features require both the limit order book and trade flow to
//! compute, capturing the interaction between resting liquidity and
//! aggressive order flow.

use crate::features::{Feature, FeatureCategory, FeatureError, MarketConfig};
use convective_data::utils;
// use atelier_data::{snapshot::MarketSnapshot};
use std::any::Any;

// ---------------------------------------------------------------------------
// PriceImpact
// ---------------------------------------------------------------------------

/// Average price impact of trades relative to the midprice:
///   Σ(trade_price - midprice) / n_trades
///
/// Positive ⇒ trades executed above mid (buying pressure),
/// negative ⇒ below mid (selling pressure). Measures how much
/// aggressive flow moves the price from the "fair" mid.
#[derive(Debug, Clone)]
pub struct PriceImpactFeature;

impl Feature for PriceImpactFeature {
    type Input = MarketSnapshot;
    type Output = f64;
    type Config = MarketConfig;

    fn name(&self) -> &'static str {
        "price_impact"
    }

    fn description(&self) -> &'static str {
        "Mean trade price deviation from midprice"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Liquidity
    }

    fn compute(
        &self,
        snap: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        let ob = snap
            .orderbook
            .as_ref()
            .ok_or(FeatureError::EmptyOrderbook)?;

        if ob.bids.is_empty() || ob.asks.is_empty() {
            return Err(FeatureError::EmptyOrderbook);
        }

        if snap.trades.is_empty() {
            return Ok(0.0);
        }

        let mid = (ob.bids[0].price + ob.asks[0].price) / 2.0;

        let total_impact: f64 = snap.trades.iter().map(|t| t.price - mid).sum();

        let avg = total_impact / snap.trades.len() as f64;
        Ok(utils::truncate_to_decimal(avg, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// TradeFlowToxicity
// ---------------------------------------------------------------------------

/// Trade flow toxicity (VPIN-inspired):
///   |buy_volume - sell_volume| / total_volume
///
/// Range [0, 1]. Values close to 1 indicate highly directional (informed)
/// flow — one side dominates. Values close to 0 indicate balanced flow.
/// This is a simplified Volume-Synchronized Probability of Informed
/// Trading (VPIN) metric.
#[derive(Debug, Clone)]
pub struct TradeFlowToxicityFeature;

impl Feature for TradeFlowToxicityFeature {
    type Input = MarketSnapshot;
    type Output = f64;
    type Config = MarketConfig;

    fn name(&self) -> &'static str {
        "trade_flow_toxicity"
    }

    fn description(&self) -> &'static str {
        "VPIN-inspired toxicity: |buy_vol - sell_vol| / total_vol"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Flow
    }

    fn compute(
        &self,
        snap: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if snap.trades.is_empty() {
            return Ok(0.0);
        }

        let mut buy_vol = 0.0_f64;
        let mut sell_vol = 0.0_f64;

        for t in &snap.trades {
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

        let toxicity = (buy_vol - sell_vol).abs() / total;
        Ok(utils::truncate_to_decimal(toxicity, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
