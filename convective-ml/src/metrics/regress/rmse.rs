use crate::metrics::interface::{MetricClass, MetricUsage, MetricValue};

#[derive(Debug)]
pub struct Rmse {
    pub id: String,
    pub values: Vec<MetricValue>,
}

impl Default for Rmse {
    fn default() -> Self {
        Self::new()
    }
}

impl Rmse {
    pub fn new() -> Self {
        Rmse {
            id: "accuracy".to_string(),
            values: Vec::new(),
        }
    }
}

impl MetricClass for Rmse {
    fn id(&self) -> &str {
        &self.id
    }

    fn metric_usage(&self) -> MetricUsage {
        MetricUsage::Class
    }

    fn compute(
        &self,
        y_true: &[f64],
        y_hat: &[f64],
        _threshold: Option<f64>,
    ) -> MetricValue {
        let n = y_true.len() as f64;
        let sum_squared_error: f64 = y_true
            .iter()
            .zip(y_hat.iter())
            .map(|(&y_true, &y_hat)| (y_hat - y_true).powf(2.0))
            .sum();

        let rmse = (sum_squared_error / n).sqrt();
        MetricValue::Scalar(rmse)
    }

    fn update(&mut self, value: MetricValue) {
        self.values.push(value);
    }

    fn latest(&self) -> Option<&MetricValue> {
        self.values.last()
    }

    fn history(&self) -> &Vec<MetricValue> {
        &self.values
    }

    fn reset(&mut self) {
        self.values.clear();
    }
}
