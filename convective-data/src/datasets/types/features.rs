use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub enum Features {
    OB,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FeatureConfig {
    pub id: Option<String>,
    pub label: Option<Features>,
    pub description: Option<String>,
    pub params_labels: Option<Vec<String>>,
    pub params_values: Option<Vec<f64>>,
}

impl FeatureConfig {
    pub fn builder() -> FeatureConfigBuilder {
        FeatureConfigBuilder::new()
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct FeatureConfigBuilder {
    pub id: Option<String>,
    pub label: Option<Features>,
    pub description: Option<String>,
    pub params_labels: Option<Vec<String>>,
    pub params_values: Option<Vec<f64>>,
}

impl Default for FeatureConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureConfigBuilder {
    pub fn new() -> Self {
        FeatureConfigBuilder {
            id: None,
            label: None,
            description: None,
            params_labels: None,
            params_values: None,
        }
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    pub fn label(mut self, label: Features) -> Self {
        self.label = Some(label);
        self
    }

    pub fn description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn params_labels(mut self, params_labels: Vec<String>) -> Self {
        self.params_labels = Some(params_labels);
        self
    }

    pub fn params_values(mut self, params_values: Vec<f64>) -> Self {
        self.params_values = Some(params_values);
        self
    }

    pub fn build(self) -> Result<FeatureConfig, &'static str> {
        let id = self.id.ok_or("Missing Feature's id")?;
        let label = self.label.ok_or("Missing Features's label")?;
        let description = self.description.ok_or("Missing Feature's description")?;
        let params_labels = self
            .params_labels
            .ok_or("Missing Features's params_labels")?;
        let params_values = self
            .params_values
            .ok_or("Missing Features's params_values")?;

        Ok(FeatureConfig {
            id: Some(id),
            label: Some(label),
            description: Some(description),
            params_labels: Some(params_labels),
            params_values: Some(params_values),
        })
    }
}
