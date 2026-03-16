use serde::Deserialize;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelProvider {
    Ollama(String),
    Xai(String),
}

impl<'de> Deserialize<'de> for ModelProvider {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        ModelProvider::from_str(&s).map_err(serde::de::Error::custom)
    }
}

impl FromStr for ModelProvider {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (provider, model) = s.split_once(':').ok_or_else(|| {
            format!("invalid model provider format: '{s}', expected 'provider:model'")
        })?;
        match provider.to_lowercase().as_str() {
            "ollama" => Ok(ModelProvider::Ollama(model.to_string())),
            "xai" => Ok(ModelProvider::Xai(model.to_string())),
            _ => Err(format!("unknown provider: '{provider}'")),
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct FileConfig {
    pub model: ModelConfig,

    #[serde(default)]
    pub experiment: ExperimentConfig,

    #[serde(default)]
    pub paths: PathsConfig,
}

#[derive(Deserialize, Debug)]
pub struct ModelConfig {
    pub embedding: ModelProvider,
    pub experiment: ModelProvider,
    pub hypothesis: ModelProvider,
}

#[derive(Deserialize, Debug)]
pub struct ExperimentConfig {
    #[serde(default = "default_max_runs")]
    pub max_runs: usize,
    #[serde(default = "default_max_seconds")]
    pub max_seconds: f64,
    #[serde(default = "default_prompt_sys")]
    pub prompt_sys: String,
    #[serde(default)]
    pub prompt_user: Option<String>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            max_runs: default_max_runs(),
            max_seconds: default_max_seconds(),
            prompt_sys: default_prompt_sys(),
            prompt_user: None,
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct PathsConfig {
    #[serde(default = "default_sys_hypothesis")]
    pub sys_hypothesis: String,
    #[serde(default = "default_user_hypothesis")]
    pub user_hypothesis: String,
    #[serde(default = "default_results_log")]
    pub results_log: String,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            sys_hypothesis: default_sys_hypothesis(),
            user_hypothesis: default_user_hypothesis(),
            results_log: default_results_log(),
        }
    }
}

fn default_max_runs() -> usize {
    10
}

fn default_max_seconds() -> f64 {
    40.0
}

fn default_prompt_sys() -> String {
    "prompt_variation.txt".to_string()
}

fn default_sys_hypothesis() -> String {
    "sys_hypothesis.txt".to_string()
}

fn default_user_hypothesis() -> String {
    "user_hypothesis_template.txt".to_string()
}

fn default_results_log() -> String {
    "results_log.tsv".to_string()
}
