//! Microprice — a fair-value estimator based on best-level imbalance.
//!
//! Unlike weighted midprice (which weights *prices* by opposing volume),
//! microprice weights each side's *price* by the opposing side's *size*:
//!
//!   microprice = bid_price × (ask_size / (bid_size + ask_size))
//!              + ask_price × (bid_size / (bid_size + ask_size))
//!
//! This pulls the estimate toward the side with *less* resting liquidity,
//! reflecting the idea that the thinner side is more likely to be consumed.

use crate::features::{Feature, FeatureCategory, FeatureError, OrderbookConfig};
use atelier_data::{datasets, orderbooks::Orderbook};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct MicropriceFeature;

impl Feature for MicropriceFeature {
    type Input = Orderbook;
    type Output = f64;
    type Config = OrderbookConfig;

    fn name(&self) -> &'static str {
        "microprice"
    }

    fn description(&self) -> &'static str {
        "Microprice: size-imbalance-weighted fair value"
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

        let bid_price = ob.bids[0].price;
        let ask_price = ob.asks[0].price;
        let bid_size = ob.bids[0].volume;
        let ask_size = ob.asks[0].volume;

        let total_size = bid_size + ask_size;
        if total_size == 0.0 {
            return Err(FeatureError::ZeroVolume);
        }

        let microprice =
            bid_price * (ask_size / total_size) + ask_price * (bid_size / total_size);
        Ok(datasets::truncate_to_decimal(microprice, 8))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
