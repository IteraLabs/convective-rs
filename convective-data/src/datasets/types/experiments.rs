use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct ExpConfig {
    pub id: String,
    pub n_progressions: u32,
    pub n_agents: Option<u32>,
}
