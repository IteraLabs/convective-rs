use crate::{
    datasets::types::{
        experiments::ExpConfig, features::FeatureConfig, models::ModelConfig,
    },
    errors::configs,
};
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub experiments: Vec<ExpConfig>,
    pub features: Option<Vec<FeatureConfig>>,
    pub models: Option<Vec<ModelConfig>>,
}

impl Config {
    pub fn load_from_toml(file_route: &str) -> Result<Self, configs::ConfigError> {
        let contents = fs::read_to_string(file_route)?;
        let config = toml::from_str(&contents)?;
        Ok(config)
    }
}
