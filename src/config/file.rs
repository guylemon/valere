use serde::Deserialize;

const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModelSpec {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub base_url: Option<String>,
}

impl ModelSpec {
    pub fn ollama_base_url(&self) -> String {
        self.base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string())
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
    pub embedding: ModelSpec,
    pub experiment: ModelSpec,
    pub hypothesis: ModelSpec,
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
