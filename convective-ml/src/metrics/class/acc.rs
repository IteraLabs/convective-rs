use crate::metrics::interface::{MetricClass, MetricUsage, MetricValue};

#[derive(Debug)]
pub struct Accuracy {
    pub id: String,
    pub values: Vec<MetricValue>,
}

impl Default for Accuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl Accuracy {
    pub fn new() -> Self {
        Accuracy {
            id: "accuracy".to_string(),
            values: Vec::new(),
        }
    }
}

impl MetricClass for Accuracy {
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
        threshold: Option<f64>,
    ) -> MetricValue {
        let threshold = threshold.unwrap_or(0.5);
        let y_true_count = y_true
            .into_iter()
            .filter(|&&values| values > threshold)
            .count();
        let y_hat_count = y_hat
            .into_iter()
            .filter(|&&values| values > threshold)
            .count();
        let y_len = y_true.len() + y_hat.len();
        let cm_acc = ((y_true_count + y_hat_count) / y_len) as f64;
        MetricValue::Scalar(cm_acc)
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
