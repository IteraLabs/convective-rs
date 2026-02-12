use crate::features::{FeatureError, FeatureSelector, OrderbookConfig};
use atelier_data::orderbooks::Orderbook;

#[derive(Debug, Clone, Copy)]
pub enum FeaturesOutput {
    Values,
    HashMap,
}

pub fn compute_features(
    orderbooks: &[Orderbook],
    feature_names: &[&str],
    depth: usize,
    bps: f64,
    _output_format: FeaturesOutput,
) -> Result<Vec<Vec<f64>>, FeatureError> {
    let selector = FeatureSelector::new(feature_names)?;
    let config = OrderbookConfig { depth, bps };

    let mut feature_matrix = Vec::new();

    for ob in orderbooks {
        let features = selector.compute_values(ob, &config)?;
        feature_matrix.push(features);
    }

    Ok(feature_matrix)
}

pub fn compute_features_with_config(
    orderbooks: &[Orderbook],
    feature_names: &[&str],
    config: &OrderbookConfig,
) -> Result<Vec<Vec<f64>>, FeatureError> {
    let selector = FeatureSelector::new(feature_names)?;

    orderbooks
        .iter()
        .map(|ob| selector.compute_values(ob, config))
        .collect()
}

// Convenience function for single orderbook
pub fn compute_single_orderbook(
    ob: &Orderbook,
    feature_names: &[&str],
    config: &OrderbookConfig,
) -> Result<Vec<f64>, FeatureError> {
    let selector = FeatureSelector::new(feature_names)?;
    selector.compute_values(ob, config)
}
