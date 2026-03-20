mod cli;
mod file;

use clap::Parser;
pub use cli::CliArgs;
pub use file::{FileConfig, ModelSpec};

use crate::error::WorkflowError;
use log::info;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Config {
    pub dry_run: bool,
    pub validate_only: bool,
    pub embedding_model: ModelSpec,
    pub experiment_max_runs: usize,
    pub experiment_max_seconds: f64,
    pub experiment_model: ModelSpec,
    pub experiment_prompt_sys_path: String,
    pub experiment_prompt_user_path: Option<String>,
    pub hypothesis_model: ModelSpec,
    pub hypothesis_prompt_sys_path: String,
    pub hypothesis_prompt_user_path: String,
    pub workflow_history_log_path: String,
}

impl Config {
    pub fn load() -> Result<Self, WorkflowError> {
        let cli = CliArgs::parse();

        let file_config = Self::load_file_config(&cli)?;

        let config = Self {
            dry_run: cli.dry_run,
            validate_only: cli.validate,
            embedding_model: file_config.model.embedding,
            experiment_model: file_config.model.experiment,
            hypothesis_model: file_config.model.hypothesis,
            experiment_max_runs: cli.max_runs.unwrap_or(file_config.experiment.max_runs),
            experiment_max_seconds: cli
                .max_seconds
                .unwrap_or(file_config.experiment.max_seconds),
            experiment_prompt_sys_path: file_config.experiment.prompt_sys,
            experiment_prompt_user_path: file_config.experiment.prompt_user,
            hypothesis_prompt_sys_path: file_config.paths.sys_hypothesis,
            hypothesis_prompt_user_path: file_config.paths.user_hypothesis,
            workflow_history_log_path: file_config.paths.results_log,
        };

        config.validate()?;

        if config.dry_run {
            info!("Dry-run mode enabled: will skip experiment execution and git commits.");
        }

        if config.validate_only {
            info!("Validation passed: configuration is valid.");
        }

        Ok(config)
    }

    fn load_file_config(cli: &CliArgs) -> Result<FileConfig, WorkflowError> {
        if let Some(path) = &cli.config {
            let content = std::fs::read_to_string(path)?;
            let config: FileConfig = toml::from_str(&content)?;
            info!("Loaded config from {:?}", path);
            return Ok(config);
        }

        if let Ok(content) = std::fs::read_to_string("valere.toml") {
            let config: FileConfig = toml::from_str(&content)?;
            info!("Loaded config from valere.toml");
            return Ok(config);
        }

        Err(WorkflowError::Config(
            "No configuration file found. Please provide a valere.toml or use --config."
                .to_string(),
        ))
    }

    fn validate(&self) -> Result<(), WorkflowError> {
        if self.embedding_model.model.is_empty() {
            return Err(WorkflowError::Config(
                "embedding model name cannot be empty".to_string(),
            ));
        }

        if self.experiment_model.model.is_empty() {
            return Err(WorkflowError::Config(
                "experiment model name cannot be empty".to_string(),
            ));
        }

        if self.hypothesis_model.model.is_empty() {
            return Err(WorkflowError::Config(
                "hypothesis model name cannot be empty".to_string(),
            ));
        }

        Self::validate_provider("embedding", &self.embedding_model.provider)?;
        Self::validate_provider("experiment", &self.experiment_model.provider)?;
        Self::validate_provider("hypothesis", &self.hypothesis_model.provider)?;

        if self.experiment_max_runs == 0 {
            return Err(WorkflowError::Config(
                "experiment max runs must be greater than 0".to_string(),
            ));
        }
        if self.experiment_max_seconds <= 0.0 {
            return Err(WorkflowError::Config(
                "experiment max seconds must be greater than 0".to_string(),
            ));
        }

        Self::validate_path(
            "experiment_prompt_sys_path",
            &self.experiment_prompt_sys_path,
        )?;
        Self::validate_path(
            "hypothesis_prompt_sys_path",
            &self.hypothesis_prompt_sys_path,
        )?;
        Self::validate_path(
            "hypothesis_prompt_user_path",
            &self.hypothesis_prompt_user_path,
        )?;

        Ok(())
    }

    fn validate_path(field_name: &str, path: &str) -> Result<(), WorkflowError> {
        if !std::path::Path::new(path).exists() {
            return Err(WorkflowError::Config(format!(
                "{} does not exist: {}",
                field_name, path
            )));
        }
        Ok(())
    }

    fn validate_provider(model_type: &str, provider: &str) -> Result<(), WorkflowError> {
        match provider.to_lowercase().as_str() {
            "ollama" | "xai" => Ok(()),
            _ => Err(WorkflowError::Config(format!(
                "invalid provider '{}' for {} model (expected 'ollama' or 'xai')",
                provider, model_type
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_valid_config(spec: ModelSpec, temp_dir: &TempDir) -> Config {
        let prompt_sys = temp_dir.path().join("prompt_variation.txt");
        let sys_hypothesis = temp_dir.path().join("sys_hypothesis.txt");
        let user_hypothesis = temp_dir.path().join("user_hypothesis.txt");

        fs::write(&prompt_sys, "").unwrap();
        fs::write(&sys_hypothesis, "").unwrap();
        fs::write(&user_hypothesis, "").unwrap();

        Config {
            dry_run: false,
            validate_only: false,
            embedding_model: spec.clone(),
            experiment_model: spec.clone(),
            hypothesis_model: spec,
            experiment_max_runs: 10,
            experiment_max_seconds: 40.0,
            experiment_prompt_sys_path: prompt_sys.to_string_lossy().to_string(),
            experiment_prompt_user_path: None,
            hypothesis_prompt_sys_path: sys_hypothesis.to_string_lossy().to_string(),
            hypothesis_prompt_user_path: user_hypothesis.to_string_lossy().to_string(),
            workflow_history_log_path: "results_log.tsv".to_string(),
        }
    }

    #[test]
    fn test_valid_ollama_provider() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_valid_config(
            ModelSpec {
                provider: "ollama".to_string(),
                model: "llama3.2:1b".to_string(),
                base_url: None,
            },
            &temp_dir,
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_valid_xai_provider() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_valid_config(
            ModelSpec {
                provider: "xai".to_string(),
                model: "grok-4-1-fast-reasoning".to_string(),
                base_url: None,
            },
            &temp_dir,
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_empty_embedding_model_name() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_valid_config(
            ModelSpec {
                provider: "ollama".to_string(),
                model: "test".to_string(),
                base_url: None,
            },
            &temp_dir,
        );
        let config = Config {
            embedding_model: ModelSpec {
                provider: "ollama".to_string(),
                model: "".to_string(),
                base_url: None,
            },
            experiment_model: config.experiment_model,
            hypothesis_model: config.hypothesis_model,
            ..config
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("embedding model name cannot be empty"));
    }

    #[test]
    fn test_empty_experiment_model_name() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_valid_config(
            ModelSpec {
                provider: "ollama".to_string(),
                model: "test".to_string(),
                base_url: None,
            },
            &temp_dir,
        );
        let config = Config {
            embedding_model: config.embedding_model,
            experiment_model: ModelSpec {
                provider: "ollama".to_string(),
                model: "".to_string(),
                base_url: None,
            },
            hypothesis_model: config.hypothesis_model,
            ..config
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("experiment model name cannot be empty"));
    }

    #[test]
    fn test_empty_hypothesis_model_name() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_valid_config(
            ModelSpec {
                provider: "ollama".to_string(),
                model: "test".to_string(),
                base_url: None,
            },
            &temp_dir,
        );
        let config = Config {
            embedding_model: config.embedding_model,
            experiment_model: config.experiment_model,
            hypothesis_model: ModelSpec {
                provider: "ollama".to_string(),
                model: "".to_string(),
                base_url: None,
            },
            ..config
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("hypothesis model name cannot be empty"));
    }

    #[test]
    fn test_invalid_provider_value() {
        let temp_dir = TempDir::new().unwrap();
        let prompt_sys = temp_dir.path().join("prompt_variation.txt");
        let sys_hypothesis = temp_dir.path().join("sys_hypothesis.txt");
        let user_hypothesis = temp_dir.path().join("user_hypothesis.txt");

        fs::write(&prompt_sys, "").unwrap();
        fs::write(&sys_hypothesis, "").unwrap();
        fs::write(&user_hypothesis, "").unwrap();

        let toml_str = format!(
            r#"
[model]
embedding = {{ provider = "invalid_provider", model = "model" }}
experiment = {{ provider = "ollama", model = "llama3.2:1b" }}
hypothesis = {{ provider = "xaii", model = "grok-4" }}

[experiment]
max_runs = 10
max_seconds = 40.0
prompt_sys = "{}"
"#,
            prompt_sys.display()
        );
        let file_config: FileConfig = toml::from_str(&toml_str).unwrap();
        let config = Config {
            dry_run: false,
            validate_only: false,
            embedding_model: file_config.model.embedding,
            experiment_model: file_config.model.experiment,
            hypothesis_model: file_config.model.hypothesis,
            experiment_max_runs: file_config.experiment.max_runs,
            experiment_max_seconds: file_config.experiment.max_seconds,
            experiment_prompt_sys_path: file_config.experiment.prompt_sys,
            experiment_prompt_user_path: file_config.experiment.prompt_user,
            hypothesis_prompt_sys_path: sys_hypothesis.to_string_lossy().to_string(),
            hypothesis_prompt_user_path: user_hypothesis.to_string_lossy().to_string(),
            workflow_history_log_path: "results_log.tsv".to_string(),
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid provider"));
    }

    #[test]
    fn test_missing_model_config() {
        let toml_str = r#"
[experiment]
max_runs = 10
max_seconds = 40.0
prompt_sys = "prompt_variation.txt"
"#;
        let result: Result<FileConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_embedding_provider() {
        let toml_str = r#"
[model]
experiment = { provider = "ollama", model = "llama3.2:1b" }
hypothesis = { provider = "xai", model = "grok-4" }
"#;
        let result: Result<FileConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_experiment_provider() {
        let toml_str = r#"
[model]
embedding = { provider = "ollama", model = "llama3.2:1b" }
hypothesis = { provider = "xai", model = "grok-4" }
"#;
        let result: Result<FileConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_hypothesis_provider() {
        let toml_str = r#"
[model]
embedding = { provider = "ollama", model = "llama3.2:1b" }
experiment = { provider = "xai", model = "grok-4" }
"#;
        let result: Result<FileConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }
}
