#[derive(Debug)]
pub struct Metrics {
    pub metrics: Vec<f64>,
    pub threshold: f64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            metrics: Vec::new(),
            threshold: 0.5,
        }
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Metrics {
            metrics: Vec::new(),
            threshold,
        }
    }

    pub fn add_metric(&mut self, metric: f64) {
        self.metrics.push(metric);
    }

    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
}
