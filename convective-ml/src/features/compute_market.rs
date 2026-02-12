//! Multi-source feature computation over [`MarketSnapshot`].
//!
//! This module parallels `compute.rs` (which operates on `Vec<Orderbook>`)
//! but accepts `MarketSnapshot` input, enabling computation of all 15
//! features from a single unified data structure.

use crate::features::{
    Feature, FeatureError, MarketConfig, OrderbookConfig,
    composite::{PriceImpactFeature, TradeFlowToxicityFeature},
    funding::FundingRateFeature,
    liquidations::{LiquidationImbalanceFeature, LiquidationPressureFeature},
    open_interest::OIChangeFeature,
    orderbook::*,
    trades::{TradeDirectionImbalanceFeature, TradeIntensityFeature},
};
use atelier_data::snapshot::MarketSnapshot;

/// Compute all 15 features for a sequence of [`MarketSnapshot`]s.
///
/// Returns a feature matrix where each row corresponds to one snapshot
/// and columns follow the canonical feature order:
///
/// 0. spread
/// 1. midprice
/// 2. w_midprice
/// 3. microprice
/// 4. vwap
/// 5. tav
/// 6. imb
/// 7. trade_intensity
/// 8. trade_direction_imbalance
/// 9. liquidation_pressure
/// 10. liquidation_imbalance
/// 11. funding_rate
/// 12. oi_change
/// 13. price_impact
/// 14. trade_flow_toxicity
///
/// Missing data sources produce 0.0 (graceful degradation).
pub fn compute_all_features(
    snapshots: &[MarketSnapshot],
    config: &MarketConfig,
) -> Result<Vec<Vec<f64>>, FeatureError> {
    // Pre-instantiate features (zero-size structs, no heap alloc)
    let spread = SpreadFeature;
    let midprice = MidpriceFeature;
    let w_midprice = WeightedMidpriceFeature;
    let microprice = MicropriceFeature;
    let vwap = VWAPFeature;
    let tav = TAVFeature;
    let imb = ImbalanceFeature;
    let trade_intensity = TradeIntensityFeature;
    let trade_dir_imb = TradeDirectionImbalanceFeature;
    let liq_pressure = LiquidationPressureFeature;
    let liq_imb = LiquidationImbalanceFeature;
    let funding = FundingRateFeature;
    let oi_change = OIChangeFeature;
    let price_impact = PriceImpactFeature;
    let toxicity = TradeFlowToxicityFeature;

    let ob_config = OrderbookConfig {
        depth: config.depth,
        bps: config.bps,
    };
    let market_config = config.clone();

    let mut prev_oi: Option<f64> = None;
    let mut matrix = Vec::with_capacity(snapshots.len());

    for snap in snapshots {
        let mut row = Vec::with_capacity(15);

        // --- Orderbook features (0-6) ---
        if let Some(ob) = &snap.orderbook {
            row.push(spread.compute(ob, &ob_config).unwrap_or(0.0));
            row.push(midprice.compute(ob, &ob_config).unwrap_or(0.0));
            row.push(w_midprice.compute(ob, &ob_config).unwrap_or(0.0));
            row.push(microprice.compute(ob, &ob_config).unwrap_or(0.0));
            row.push(vwap.compute(ob, &ob_config).unwrap_or(0.0));
            row.push(tav.compute(ob, &ob_config).unwrap_or(0.0));
            row.push(imb.compute(ob, &ob_config).unwrap_or(0.0));
        } else {
            row.extend_from_slice(&[0.0; 7]);
        }

        // --- Trade features (7-8) ---
        row.push(
            trade_intensity
                .compute(&snap.trades, &market_config)
                .unwrap_or(0.0),
        );
        row.push(
            trade_dir_imb
                .compute(&snap.trades, &market_config)
                .unwrap_or(0.0),
        );

        // --- Liquidation features (9-10) ---
        row.push(
            liq_pressure
                .compute(&snap.liquidations, &market_config)
                .unwrap_or(0.0),
        );
        row.push(
            liq_imb
                .compute(&snap.liquidations, &market_config)
                .unwrap_or(0.0),
        );

        // --- Funding rate (11) ---
        if let Some(fr) = &snap.funding_rate {
            row.push(funding.compute(fr, &market_config).unwrap_or(0.0));
        } else {
            row.push(0.0);
        }

        // --- OI change (12) ---
        if let Some(oi) = &snap.open_interest {
            let curr = oi.open_interest;
            let prev = prev_oi.unwrap_or(curr);
            row.push(
                oi_change
                    .compute(&[prev, curr], &market_config)
                    .unwrap_or(0.0),
            );
            prev_oi = Some(curr);
        } else {
            row.push(0.0);
        }

        // --- Composite features (13-14) ---
        row.push(price_impact.compute(snap, &market_config).unwrap_or(0.0));
        row.push(toxicity.compute(snap, &market_config).unwrap_or(0.0));

        matrix.push(row);
    }

    Ok(matrix)
}

/// Names of all 15 features in canonical order.
pub const ALL_FEATURE_NAMES: [&str; 15] = [
    "spread",
    "midprice",
    "w_midprice",
    "microprice",
    "vwap",
    "tav",
    "imb",
    "trade_intensity",
    "trade_direction_imbalance",
    "liquidation_pressure",
    "liquidation_imbalance",
    "funding_rate",
    "oi_change",
    "price_impact",
    "trade_flow_toxicity",
];
