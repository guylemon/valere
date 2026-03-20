mod result;

use crate::config::ModelSpec;
use crate::util::read_workflow_input_file;
use crate::{config::Config, error::WorkflowError};
use llm_msg::{Message, Role};
use llm_provider::{ChatRequest, Provider as LlmProvider};
use log::{debug, info, warn};
use std::time::Instant;

pub use result::{ExperimentResult, ExperimentResultBuilder};

pub struct Experiment {
    pub result: ExperimentResult,
    total_queries: usize,
    total_relevance: f64,
    valid_output_count: usize,
    llm_provider: llm_provider::Provider,
    embedding_provider: llm_embed::Provider,
}

impl Experiment {
    pub fn new(configuration: &Config) -> Self {
        let llm_provider = llm_provider(&configuration.experiment_model);
        let embedding_provider = embedding_provider(&configuration.embedding_model);

        Self {
            result: ExperimentResult::default(),
            total_queries: 0,
            total_relevance: 0.0,
            valid_output_count: 0,
            llm_provider,
            embedding_provider,
        }
    }

    pub fn run(&mut self, configuration: &Config) -> Result<(), WorkflowError> {
        info!("Experiment start");

        let (topics, prompt_variation) = read_experiment_input_files(configuration)?;
        let start = Instant::now();

        for topic in &topics {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > configuration.experiment_max_seconds {
                warn!("Maximum experiment seconds exceeded: {elapsed}");
                break;
            }

            let response = run_test_prompt(
                topic,
                &prompt_variation,
                &configuration.experiment_model.model,
                &self.llm_provider,
            )?;

            let generated_queries: Vec<String> = match serde_json::from_str(&response.content) {
                Ok(vec) => vec,
                Err(e) => {
                    debug!("Error parsing generated queries for {topic}.\n{e}");
                    continue;
                }
            };

            debug!("Generated queries: {generated_queries:?}");

            // TODO this needs to be an experiment parameter
            let target_length = 5;
            let num_queries = generated_queries.len();
            let embedding_provider = &self.embedding_provider;

            debug!("Validating queries for topic: {topic}");
            if num_queries == target_length {
                info!("Queries validated. Scoring relevance.");
                self.valid_output_count += 1;

                // TODO batch the embedding calls for topics and queries, and zip the cosine
                // similarity?
                // TODO need more granular error enum for instrumentum llm generation to help
                // determine transience
                // TODO need retry logic for this so it doesn't fail forever when Ollama is down.
                debug!("Generating embedding for {topic}");
                let topic_embedding =
                    match llm_embed::generate(vec![topic.to_string()], embedding_provider) {
                        Ok(embedding) => embedding,
                        Err(e) => {
                            warn!("Embedding failure: {e}");
                            continue;
                        }
                    };
                debug!("Topic embedding success for {topic}");

                for query in generated_queries {
                    debug!("Generating embedding for {}", query.clone());
                    let query_embedding = match llm_embed::generate(vec![query], embedding_provider)
                    {
                        Ok(embedding) => embedding,
                        Err(e) => {
                            warn!("Embedding failure: {e}");
                            continue;
                        }
                    };
                    debug!("Query embedding success");

                    let cosine_similarity = cosine_sim(
                        &topic_embedding.embeddings[0],
                        &query_embedding.embeddings[0],
                    );

                    self.total_relevance += cosine_similarity;
                    self.total_queries += 1;
                }
            } else {
                info!("Expected {target_length} queries but received {num_queries}.");
            }
        }

        let total_seconds = start.elapsed().as_secs_f64();
        let validity_rate = self.valid_output_count as f64 / topics.len() as f64;
        let avg_relevance = if self.total_queries > 0 {
            self.total_relevance / self.total_queries as f64
        } else {
            0.0
        };
        let score = validity_rate * avg_relevance;

        let result: ExperimentResult = ExperimentResultBuilder::new()
            .total_seconds(total_seconds)
            .validity_rate(validity_rate)
            .avg_relevance(avg_relevance)
            .score(score)
            .build();

        let summary_string = format!(
            "total seconds: {}\nvalidity rate: {}\naverage relevance: {}\nscore: {}",
            result.total_seconds, result.validity_rate, result.avg_relevance, result.score,
        );
        info!("Experiment run completed with result:\n{summary_string}");

        self.result = result;
        Ok(())
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0; // or panic in debug
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot as f64
}

fn read_experiment_input_files(
    configuration: &Config,
) -> Result<(Vec<String>, String), WorkflowError> {
    // TODO topics.json needs to be configured separately.
    let topics = read_workflow_input_file("topics.json")?;
    let prompt_variation = read_workflow_input_file(&configuration.experiment_prompt_sys_path)?;

    info!("Parsing topics");
    let topics: Vec<String> = serde_json::from_str(&topics)?;
    debug!("Topics: {topics:?}");

    Ok((topics, prompt_variation))
}

fn embedding_provider(spec: &ModelSpec) -> llm_embed::Provider {
    if spec.provider.to_lowercase() != "ollama" {
        panic!("No support for embedding providers other than Ollama at this time.");
    }
    llm_embed::Provider::Ollama {
        model: spec.model.clone(),
    }
}

fn llm_provider(spec: &ModelSpec) -> LlmProvider {
    match spec.provider.to_lowercase().as_str() {
        "ollama" => {
            let base_url = Some(format!("{}/api", spec.ollama_base_url()));
            LlmProvider::Ollama(llm_provider::Config::new(base_url.as_deref()))
        }
        "xai" => {
            let api_key = std::env::var("XAI_API_KEY").unwrap_or("no-key-provided".to_string());
            LlmProvider::Xai {
                api_key,
                model: spec.model.clone(),
            }
        }
        _ => {
            panic!("Unknown provider: {}", spec.provider);
        }
    }
}

fn run_test_prompt(
    topic: &str,
    prompt: &str,
    model_name: &str,
    provider: &LlmProvider,
) -> Result<Message, WorkflowError> {
    info!("Generating response for prompt variation. topic: {topic}");
    let messages: Vec<Message> = vec![
        Message::new(Role::System, prompt),
        Message::new(Role::User, topic),
    ];

    let chat_request = match ChatRequest::builder(model_name)
        .messages(messages)
        .options(llm_provider::Options::recommended())
        .stream(false)
        .build()
    {
        Ok(req) => req,
        Err(e) => return Err(WorkflowError::Config(e.to_string())),
    };

    match llm_generate::generate(&chat_request, provider) {
        Ok(response) => {
            debug!("Received response for topic: {topic}\n{response:?}");
            Ok(response)
        }
        Err(e) => {
            let msg = format!("Error generating response for test prompt {prompt}: {e}");
            Err(WorkflowError::LanguageModelError(msg))
        }
    }
}
