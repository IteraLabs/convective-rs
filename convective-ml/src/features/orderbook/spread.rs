use crate::features::{Feature, FeatureCategory, FeatureError, OrderbookConfig};
use atelier_data::{datasets, orderbooks::Orderbook};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct SpreadFeature;

impl Feature for SpreadFeature {
    type Input = Orderbook;
    type Output = f64;
    type Config = OrderbookConfig;

    fn name(&self) -> &'static str {
        "spread"
    }

    fn description(&self) -> &'static str {
        "Bid-ask spread (ask_price - bid_price)"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Spread
    }

    fn compute(
        &self,
        ob: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        if ob.bids.is_empty() || ob.asks.is_empty() {
            return Err(FeatureError::EmptyOrderbook);
        }

        let spread = ob.asks[0].price - ob.bids[0].price;
        Ok(datasets::truncate_to_decimal(spread, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
