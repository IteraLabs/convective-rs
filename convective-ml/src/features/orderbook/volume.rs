use crate::features::{Feature, FeatureCategory, FeatureError, OrderbookConfig};
use atelier_data::{datasets, orderbooks::Orderbook};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct VWAPFeature;

impl Feature for VWAPFeature {
    type Input = Orderbook;
    type Output = f64;
    type Config = OrderbookConfig;

    fn name(&self) -> &'static str {
        "vwap"
    }

    fn description(&self) -> &'static str {
        "Volume-Weighted Average Price up to specified depth"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Volume
    }

    fn compute(
        &self,
        ob: &Self::Input,
        config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if ob.bids.is_empty() || ob.asks.is_empty() {
            return Err(FeatureError::EmptyOrderbook);
        }

        let depth = config.depth;
        if depth > ob.bids.len() || depth > ob.asks.len() {
            return Err(FeatureError::InsufficientDepth {
                requested: depth,
                available: ob.bids.len().min(ob.asks.len()),
            });
        }

        let bid_levels = ob.bids.iter().take(depth);
        let ask_levels = ob.asks.iter().take(depth);
        let all_levels = bid_levels.chain(ask_levels);

        let (sum_p_v, sum_v) = all_levels.fold((0.0, 0.0), |(acc_p_v, acc_v), level| {
            (acc_p_v + level.price * level.volume, acc_v + level.volume)
        });

        if sum_v > 0.0 {
            Ok(datasets::truncate_to_decimal(sum_p_v / sum_v, 8))
        } else {
            Err(FeatureError::ZeroVolume)
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct TAVFeature;

impl Feature for TAVFeature {
    type Input = Orderbook;
    type Output = f64;
    type Config = OrderbookConfig;

    fn name(&self) -> &'static str {
        "tav"
    }

    fn description(&self) -> &'static str {
        "Total Available Volume within X bps of midprice"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Volume
    }

    fn compute(
        &self,
        ob: &Self::Input,
        config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if ob.bids.is_empty() || ob.asks.is_empty() {
            return Err(FeatureError::EmptyOrderbook);
        }

        let best_bid = ob.bids[0].price;
        let best_ask = ob.asks[0].price;
        let bps = config.bps;

        let upper_ask = best_ask * (1.0 + bps);
        let lower_bid = best_bid * (1.0 - bps);

        let bid_volume: f64 = ob
            .bids
            .iter()
            .filter(|level| level.price >= lower_bid)
            .map(|level| level.volume)
            .sum();

        let ask_volume: f64 = ob
            .asks
            .iter()
            .filter(|level| level.price <= upper_ask)
            .map(|level| level.volume)
            .sum();

        let tav = bid_volume + ask_volume;
        Ok(datasets::truncate_to_decimal(tav, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
