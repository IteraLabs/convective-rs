//! Interfaces for Metrics
//! `ClassMetric` : For Classification
//! `RegressMetric` : For Regression

use std::collections::HashMap;

/// The type of the metric according to the taxonomy of the model that could
/// use it.
///
/// - Classification : Categorical outputs
/// - Regression : Numeric outputs
///
#[derive(Debug, Clone)]
pub enum MetricUsage {
    Class,
    Regress,
    Multiple,
}

/// The type of the resulted value (after compute or transformation)
#[derive(Debug, Clone)]
pub enum MetricValue {
    Numeric(u64),
    Scalar(f64),
    ScalarMatrix(Vec<Vec<f64>>),
    CategoricalMatrix(Vec<Vec<String>>),
    Multiple(HashMap<String, f64>),
}

impl MetricValue {
    pub fn from_numeric(&self) -> Option<u64> {
        match self {
            MetricValue::Numeric(val) => Some(*val),
            _ => None,
        }
    }

    pub fn from_scalar(&self) -> Option<f64> {
        match self {
            MetricValue::Scalar(val) => Some(*val),
            _ => None,
        }
    }

    pub fn from_numeric_matrix(&self) -> Option<&Vec<Vec<f64>>> {
        match self {
            MetricValue::ScalarMatrix(mat) => Some(mat),
            _ => None,
        }
    }

    pub fn from_hasmap_f64(&self) -> Option<&HashMap<String, f64>> {
        match self {
            MetricValue::Multiple(mul) => Some(mul),
            _ => None,
        }
    }
}

// Metric Trait for Classification Metrics
pub trait MetricClass: std::fmt::Debug {
    fn id(&self) -> &str;
    fn metric_usage(&self) -> MetricUsage;
    fn update(&mut self, value: MetricValue);
    fn latest(&self) -> Option<&MetricValue>;
    fn history(&self) -> &Vec<MetricValue>;
    fn reset(&mut self);

    fn compute(
        &self,
        y_true: &[f64],
        y_hat: &[f64],
        threshold: Option<f64>,
    ) -> MetricValue;
}
