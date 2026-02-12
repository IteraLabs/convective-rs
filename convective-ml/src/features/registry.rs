use crate::features::FeatureCategory;
use std::{collections::HashMap, sync::RwLock};

// Simplified registry without complex type erasure for now
pub struct FeatureRegistry {
    feature_names: RwLock<HashMap<String, FeatureCategory>>,
    categories: RwLock<HashMap<FeatureCategory, Vec<String>>>,
}

impl FeatureRegistry {
    pub fn new() -> Self {
        Self {
            feature_names: RwLock::new(HashMap::new()),
            categories: RwLock::new(HashMap::new()),
        }
    }

    pub fn register_feature(&self, name: &str, category: FeatureCategory) {
        let mut names = self.feature_names.write().unwrap();
        let mut categories = self.categories.write().unwrap();

        names.insert(name.to_string(), category.clone());
        categories
            .entry(category)
            .or_default()
            .push(name.to_string());
    }

    pub fn list_features(&self) -> Vec<String> {
        let names = self.feature_names.read().unwrap();
        names.keys().cloned().collect()
    }

    pub fn list_by_category(&self, category: FeatureCategory) -> Vec<String> {
        let categories = self.categories.read().unwrap();
        categories.get(&category).cloned().unwrap_or_default()
    }

    pub fn feature_exists(&self, name: &str) -> bool {
        let names = self.feature_names.read().unwrap();
        names.contains_key(name)
    }

    pub fn get_category(&self, name: &str) -> Option<FeatureCategory> {
        let names = self.feature_names.read().unwrap();
        names.get(name).cloned()
    }
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Global registry instance
lazy_static::lazy_static! {
    pub static ref ORDERBOOK_FEATURES: FeatureRegistry = {
        let registry = FeatureRegistry::new();

        // Register all orderbook features
        registry.register_feature("spread", FeatureCategory::Spread);
        registry.register_feature("midprice", FeatureCategory::Price);
        registry.register_feature("w_midprice", FeatureCategory::Price);
        registry.register_feature("microprice", FeatureCategory::Price);
        registry.register_feature("vwap", FeatureCategory::Volume);
        registry.register_feature("imb", FeatureCategory::Imbalance);
        registry.register_feature("tav", FeatureCategory::Volume);

        registry
    };

    pub static ref TRADE_FEATURES: FeatureRegistry = {
        let registry = FeatureRegistry::new();
        registry.register_feature("trade_intensity", FeatureCategory::Flow);
        registry.register_feature("trade_direction_imbalance", FeatureCategory::Flow);
        registry
    };

    pub static ref LIQUIDATION_FEATURES: FeatureRegistry = {
        let registry = FeatureRegistry::new();
        registry.register_feature("liquidation_pressure", FeatureCategory::Flow);
        registry.register_feature("liquidation_imbalance", FeatureCategory::Imbalance);
        registry
    };

    pub static ref MARKET_FEATURES: FeatureRegistry = {
        let registry = FeatureRegistry::new();
        // Single-source features that use specific inputs
        registry.register_feature("funding_rate", FeatureCategory::Flow);
        registry.register_feature("oi_change", FeatureCategory::Volume);
        // Composite features that combine orderbook + trades
        registry.register_feature("price_impact", FeatureCategory::Liquidity);
        registry.register_feature("trade_flow_toxicity", FeatureCategory::Flow);
        registry
    };
}
