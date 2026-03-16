use crate::error::WorkflowError;
use crate::util::{read_history_log_first_two_columns, read_workflow_input_file, LogRecord};
use crate::Config;
use llm_generate::Message;
use llm_generate::Provider;
use log::{debug, info, warn};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

use csv::ReaderBuilder;

/// A hypothesis represents a recommendation from the hypothesizer assistant
#[derive(Debug, Default, Deserialize)]
pub struct Hypothesis {
    /// A short description of the prompt change to test
    pub description: String,
    /// The modified prompt
    pub prompt: String,
}

#[derive(Debug, Default)]
pub struct Proposer {
    pub hypothesis: Hypothesis,
}

impl Proposer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(&mut self, configuration: &Config) -> Result<(), WorkflowError> {
        checkout_last_good_prompt(configuration)?;
        let system_prompt = read_workflow_input_file(&configuration.hypothesis_prompt_sys_path)?;
        let user_prompt = render_proposer_user_prompt(configuration)?;
        let provider = provider(&configuration.hypothesis_provider)?;

        self.hypothesis =
            generate_hypothesis(system_prompt, user_prompt, provider).and_then(parse_hypothesis)?;

        write_config_files(&configuration.experiment_prompt_sys_path, &self.hypothesis)
    }
}

fn provider(hypothesis_provider: &crate::config::ModelProvider) -> Result<Provider, WorkflowError> {
    match hypothesis_provider {
        crate::config::ModelProvider::Ollama(model) => Ok(Provider::Ollama {
            model: model.to_string(),
        }),
        crate::config::ModelProvider::Xai(model) => {
            let api_key = match std::env::var("XAI_API_KEY") {
                Ok(k) => k,
                Err(_) => {
                    return Err(WorkflowError::Config(
                        "XAI_API_KEY is required to use XAI provider".to_string(),
                    ));
                }
            };
            Ok(Provider::Xai {
                api_key,
                model: model.to_string(),
            })
        }
    }
}

fn render_proposer_user_prompt(configuration: &Config) -> Result<String, WorkflowError> {
    info!("Rendering proposer user prompt template.");
    let template = read_workflow_input_file(&configuration.hypothesis_prompt_user_path)?;
    let history = read_history_log_first_two_columns(&configuration.workflow_history_log_path)?;
    let current_prompt = read_workflow_input_file(&configuration.experiment_prompt_sys_path)?;

    let mut variables = HashMap::new();
    variables.insert("history".to_string(), history);
    variables.insert("current_prompt".to_string(), current_prompt);

    let rendered_prompt = llm_prompt::substitute(&template, &variables)?;
    debug!("Rendered template: {rendered_prompt}");

    Ok(rendered_prompt)
}

fn generate_hypothesis(
    system_prompt: String,
    user_prompt: String,
    provider: Provider,
) -> Result<String, WorkflowError> {
    info!("Generating hypothesis.");

    let system_prompt = Message::new(llm_msg::Role::System, &system_prompt);
    let user_prompt = Message::new(llm_msg::Role::User, &user_prompt);
    let messages = vec![system_prompt, user_prompt];
    let tools_enabled = false;

    match llm_generate::generate(messages, tools_enabled, &provider) {
        Ok(response) => {
            let hypothesis = response.content;
            debug!("Generation succeeded. Raw hypothesis:\n{hypothesis}");

            Ok(hypothesis)
        }
        Err(e) => Err(WorkflowError::LanguageModelError(e.to_string())),
    }
}

fn parse_hypothesis(raw_hypothesis: String) -> Result<Hypothesis, WorkflowError> {
    info!("Parsing hypothesis from LLM response.");
    match serde_json::from_str::<Hypothesis>(&raw_hypothesis) {
        Ok(hypothesis) => {
            info!("Hypothesis parsing succeeded.");
            debug!("{hypothesis:?}");

            Ok(hypothesis)
        }
        _ => Err(WorkflowError::LanguageModelError(
            "Failed to parse LLM-generated hypothesis".to_string(),
        )),
    }
}

fn write_config_files(file_path: &str, hypothesis: &Hypothesis) -> Result<(), WorkflowError> {
    let contents = &hypothesis.prompt;
    let path = &file_path;

    info!("Writing experimental prompt to {path}");
    fs::write(path, contents)?;
    info!("Experimental prompt write to {path} succeeded.");

    Ok(())
}

fn checkout_last_good_prompt(configuration: &Config) -> Result<String, WorkflowError> {
    info!("Pulling prompt for best run");

    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(&configuration.workflow_history_log_path)?;

    let mut best_prompt = String::new();
    let mut best_score = f64::NEG_INFINITY;

    for result in rdr.deserialize() {
        let record: LogRecord = result?;
        if record.score() > best_score {
            best_score = record.score();
            best_prompt = record.prompt().to_string();
        }
    }

    info!("Best prompt:\n{best_prompt}");
    if best_prompt.is_empty() {
        warn!("No best prompt found. Is this the first run?");
    }

    Ok(best_prompt)
}
