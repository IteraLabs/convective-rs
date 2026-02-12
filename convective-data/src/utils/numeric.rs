//! # convective-data :: utils :: numeric

use rust_decimal::Decimal;
use std::str::FromStr;

pub fn decimal_to_f64(d: Decimal) -> f64 {
    f64::from_str(&d.to_string()).unwrap_or(0.0)
}

/// Truncate decimals on a f64
pub fn truncate_to_decimal(num: f64, decimal_places: u32) -> f64 {
    let multiplier = 10_f64.powi(decimal_places as i32);
    (num * multiplier).trunc() / multiplier
}

