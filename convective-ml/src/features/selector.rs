use crate::features::{Feature, FeatureError, OrderbookConfig, orderbook::*};
use atelier_data::orderbooks::Orderbook;

pub struct FeatureSelector {
    features:
        Vec<Box<dyn Feature<Input = Orderbook, Output = f64, Config = OrderbookConfig>>>,
    feature_names: Vec<String>,
}

impl FeatureSelector {
    pub fn new(feature_names: &[&str]) -> Result<Self, FeatureError> {
        let mut features: Vec<
            Box<dyn Feature<Input = Orderbook, Output = f64, Config = OrderbookConfig>>,
        > = Vec::new();
        let mut names = Vec::new();

        for &name in feature_names {
            let feature: Box<
                dyn Feature<Input = Orderbook, Output = f64, Config = OrderbookConfig>,
            > = match name {
                "spread" => Box::new(SpreadFeature),
                "midprice" => Box::new(MidpriceFeature),
                "w_midprice" => Box::new(WeightedMidpriceFeature),
                "vwap" => Box::new(VWAPFeature),
                "imb" => Box::new(ImbalanceFeature),
                "tav" => Box::new(TAVFeature),
                "microprice" => Box::new(MicropriceFeature),
                _ => {
                    return Err(FeatureError::FeatureNotFound {
                        name: name.to_string(),
                    });
                }
            };

            features.push(feature);
            names.push(name.to_string());
        }

        Ok(Self {
            features,
            feature_names: names,
        })
    }

    pub fn from_features(
        features: Vec<
            Box<dyn Feature<Input = Orderbook, Output = f64, Config = OrderbookConfig>>,
        >,
    ) -> Self {
        let names = features.iter().map(|f| f.name().to_string()).collect();
        Self {
            features,
            feature_names: names,
        }
    }

    pub fn compute_values(
        &self,
        ob: &Orderbook,
        config: &OrderbookConfig,
    ) -> Result<Vec<f64>, FeatureError> {
        self.features
            .iter()
            .map(|feature| feature.compute(ob, config))
            .collect()
    }

    pub fn compute_values_with_defaults(
        &self,
        ob: &Orderbook,
    ) -> Result<Vec<f64>, FeatureError> {
        let default_config = OrderbookConfig::default();
        self.compute_values(ob, &default_config)
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}
