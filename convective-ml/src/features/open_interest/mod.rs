//! Open interest change feature.

use crate::features::{Feature, FeatureCategory, FeatureError, MarketConfig};
use std::any::Any;

// ---------------------------------------------------------------------------
// OIChangeFeature
// ---------------------------------------------------------------------------

/// Percentage change in open interest: (curr - prev) / prev × 100.
///
/// Input is a tuple `(prev_oi, curr_oi)` as `[f64; 2]`.
/// Positive ⇒ new positions entering, negative ⇒ positions closing.
#[derive(Debug, Clone)]
pub struct OIChangeFeature;

impl Feature for OIChangeFeature {
    type Input = [f64; 2];
    type Output = f64;
    type Config = MarketConfig;

    fn name(&self) -> &'static str {
        "oi_change"
    }

    fn description(&self) -> &'static str {
        "Percentage change in open interest: (curr - prev) / prev * 100"
    }

    fn category(&self) -> FeatureCategory {
        FeatureCategory::Volume
    }

    fn compute(
        &self,
        oi_pair: &Self::Input,
        _config: &Self::Config,
    ) -> Result<Self::Output, FeatureError> {
        let [prev, curr] = *oi_pair;

        if prev == 0.0 {
            // Cannot compute percentage change from zero base
            if curr == 0.0 {
                return Ok(0.0);
            }
            return Err(FeatureError::ComputationError {
                message: "previous OI is zero, cannot compute percentage change"
                    .to_string(),
            });
        }

        Ok((curr - prev) / prev * 100.0)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
