use crate::features::{Feature, FeatureCategory, FeatureError, OrderbookConfig};
use atelier_data::{datasets, orderbooks::Orderbook};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct MidpriceFeature;

impl Feature for MidpriceFeature {
    type Input = Orderbook;
    type Output = f64;
    type Config = OrderbookConfig;

    fn name(&self) -> &'static str {
        "midprice"
    }

    fn description(&self) -> &'static str {
        "Mid price: (best_bid + best_ask) / 2"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Price
    }

    fn compute(
        &self,
        ob: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if ob.bids.is_empty() || ob.asks.is_empty() {
            return Err(FeatureError::EmptyOrderbook);
        }

        let midprice = (ob.asks[0].price + ob.bids[0].price) / 2.0;
        Ok(datasets::truncate_to_decimal(midprice, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct WeightedMidpriceFeature;

impl Feature for WeightedMidpriceFeature {
    type Input = Orderbook;
    type Output = f64;
    type Config = OrderbookConfig;

    fn name(&self) -> &'static str {
        "w_midprice"
    }

    fn description(&self) -> &'static str {
        "Volume-weighted mid price at best levels"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Price
    }

    fn compute(
        &self,
        ob: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if ob.bids.is_empty() || ob.asks.is_empty() {
            return Err(FeatureError::EmptyOrderbook);
        }

        let total_volume = ob.asks[0].volume + ob.bids[0].volume;
        if total_volume == 0.0 {
            return Err(FeatureError::ZeroVolume);
        }

        let w_midprice = ((ob.bids[0].price * ob.bids[0].volume)
            + (ob.asks[0].price * ob.asks[0].volume))
            / total_volume;
        Ok(datasets::truncate_to_decimal(w_midprice, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
