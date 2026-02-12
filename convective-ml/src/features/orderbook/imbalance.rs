use crate::features::{Feature, FeatureCategory, FeatureError, OrderbookConfig};
use atelier_data::{datasets, orderbooks::Orderbook};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct ImbalanceFeature;

impl Feature for ImbalanceFeature {
    type Input = Orderbook;
    type Output = f64;
    type Config = OrderbookConfig;

    fn name(&self) -> &'static str {
        "imb"
    }

    fn description(&self) -> &'static str {
        "Order imbalance: ask_volume / (ask_volume + bid_volume)"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Imbalance
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

        let imbalance = ob.asks[0].volume / total_volume;
        Ok(datasets::truncate_to_decimal(imbalance, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
