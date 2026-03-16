use std::{
    collections::HashMap,
    fs,
    io::Write,
    process::{self, Command},
    time::Instant,
};

use llm_generate::Provider;
use llm_msg::{Message, Role};
use llm_prompt::substitute;
use log::{debug, error, info};
use serde::Deserialize;

type WorkflowError = Box<dyn std::error::Error>;

#[derive(Debug, Default)]
enum WorkflowAction {
    #[default]
    Discard,
    Keep,
}

#[derive(Default, Debug)]
struct ExperimentResult {
    total_seconds: f64,
    validity_rate: f64,
    avg_relevance: f64,
    score: f64,
}

#[derive(Default, Debug)]
struct ExperimentResultBuilder {
    total_seconds: f64,
    validity_rate: f64,
    avg_relevance: f64,
    score: f64,
}

impl ExperimentResultBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn build(self) -> ExperimentResult {
        ExperimentResult {
            total_seconds: self.total_seconds,
            validity_rate: self.validity_rate,
            avg_relevance: self.avg_relevance,
            score: self.score,
        }
    }

    fn total_seconds(mut self, total_seconds: f64) -> ExperimentResultBuilder {
        self.total_seconds = total_seconds;
        self
    }

    fn validity_rate(mut self, validity_rate: f64) -> ExperimentResultBuilder {
        self.validity_rate = validity_rate;
        self
    }

    fn avg_relevance(mut self, avg_relevance: f64) -> ExperimentResultBuilder {
        self.avg_relevance = avg_relevance;
        self
    }

    fn score(mut self, score: f64) -> ExperimentResultBuilder {
        self.score = score;
        self
    }
}

#[derive(Default, Debug)]
struct State {
    config: Config,
    results_log: String,
    sys_prompt_hypothesis: String,
    sys_prompt_variation: String,
    user_prompt_template_string: String,
    user_prompt_rendered: String,
    hypothesis_raw: String,
    hypothesis: Hypothesis,
    current_experiment_hash: String,
    experiment_current_result: ExperimentResult,
    experiment_high_score: f64,
    experiment_action: WorkflowAction,
    runs: usize,
}

impl State {
    fn new(config: Config) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }
}

#[derive(Default, Debug, Clone)]
struct Config {
    dry_run: bool,
    log_path_results: String,
    prompt_path_experiment: String,
    prompt_path_hypothesis_sys: String,
    prompt_path_hypothesis_user: String,
    max_experiment_seconds: f64,
    experiment_model: String,
    embedding_model: String,
    max_experiment_runs: usize,
}

impl Config {
    fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
}

/// A hypothesis represents a recommendation from the hypothesizer assistant
#[derive(Debug, Default, Deserialize)]
struct Hypothesis {
    /// A short description of the prompt change to test
    description: String,
    /// The modified prompt
    prompt: String,
}

macro_rules! define_state_loader {
    ($name:ident, $field:ident, $config_path:ident) => {
        fn $name(mut state: State) -> Result<State, WorkflowError> {
            let path = &state.config.$config_path;
            info!("Reading {} from {}", stringify!($field), path);
            let content = fs::read_to_string(path)?;
            debug!("Content from {path}:\n{content}");
            state.$field = content;
            Ok(state)
        }
    };
}

define_state_loader!(
    read_sys_prompt_file,
    sys_prompt_hypothesis,
    prompt_path_hypothesis_sys
);
define_state_loader!(
    read_current_prompt,
    sys_prompt_variation,
    prompt_path_experiment
);
define_state_loader!(
    read_user_prompt,
    user_prompt_template_string,
    prompt_path_hypothesis_user
);

trait PipelineExt<T> {
    fn pipe<U, F>(self, f: F) -> Result<U, WorkflowError>
    where
        F: FnOnce(T) -> Result<U, WorkflowError>;
}

impl<T> PipelineExt<T> for T {
    fn pipe<U, F>(self, f: F) -> Result<U, WorkflowError>
    where
        F: FnOnce(T) -> Result<U, WorkflowError>,
    {
        f(self)
    }
}

trait OutputExt {
    fn stdout_str(&self) -> String;
}

impl OutputExt for process::Output {
    fn stdout_str(&self) -> String {
        String::from_utf8_lossy(&self.stdout).trim().to_string()
    }
}

// RUST_LOG=debug cargo run
fn main() -> Result<(), WorkflowError> {
    env_logger::init();

    info!("Valere workflow start.");

    let mut config = Config::new();
    config.prompt_path_hypothesis_sys = "sys_hypothesis.txt".to_string();
    config.log_path_results = "results_log.tsv".to_string();
    config.prompt_path_experiment = "prompt_variation.txt".to_string();
    config.prompt_path_hypothesis_user = "user_hypothesis_template.txt".to_string();
    config.dry_run = std::env::args().any(|arg| arg == "--dry-run");
    config.max_experiment_seconds = 40.0;
    config.experiment_model = "llama3.2:1b".to_string();
    config.embedding_model = "embeddinggemma".to_string();
    config.max_experiment_runs = 10;
    let max_experiment_runs = config.max_experiment_runs;

    if config.dry_run {
        info!("Dry-run mode enabled: will skip experiment execution and git commits.");
    }

    let mut state = State::new(config);

    for run in 0..max_experiment_runs {
        let current_run = run + 1;
        let config = state.config.clone();

        let result = (|| -> Result<State, WorkflowError> {
            let mut state = State::new(config.clone());
            state.runs = current_run;

            state = state.pipe(read_sys_prompt_file)?;
            state = state.pipe(get_current_high_score)?;
            state = state.pipe(read_current_prompt)?;
            state = state.pipe(read_user_prompt)?;
            state = state.pipe(substitute_user_prompt_vars)?;
            state = state.pipe(generate_hypothesis)?;
            state = state.pipe(parse_hypothesis)?;
            state = state.pipe(write_config_files)?;
            state = state.pipe(commit_configs)?;
            state = state.pipe(run_experiment)?;
            state = state.pipe(judge_results)?;
            state = state.pipe(log_results)?;
            state = state.pipe(rollback_or_pass)?;
            Ok(state)
        })();

        match result {
            Ok(new_state) => state = new_state,
            Err(e) => {
                error!("WorkflowError in run {current_run}: {e}");
                let crash_state = State::new(config.clone());
                let _ = log_crash(&crash_state, &format!("crash:{}", e));
                if crash_state.current_experiment_hash.is_empty() {
                    // No commit was made yet
                } else {
                    reset_git_state();
                }
                continue;
            }
        }
    }

    Ok(())
}

fn substitute_user_prompt_vars(mut state: State) -> Result<State, WorkflowError> {
    info!("Rendering user prompt template.");
    let mut variables = HashMap::new();
    variables.insert("history".to_string(), state.results_log.clone());
    variables.insert(
        "current_prompt".to_string(),
        state.sys_prompt_variation.clone(),
    );

    let rendered_prompt = substitute(&state.user_prompt_template_string, &variables)?;
    debug!("Rendered template: {rendered_prompt}");

    state.user_prompt_rendered = rendered_prompt.trim().to_string();
    Ok(state)
}

fn generate_hypothesis(mut state: State) -> Result<State, WorkflowError> {
    info!("Configuring LLM call for hypothesis generation.");
    let api_key = std::env::var("XAI_API_KEY")?;
    let model = "grok-4-1-fast-reasoning".to_string();
    let provider = Provider::Xai { api_key, model };
    let system_prompt = Message::new(llm_msg::Role::System, &state.sys_prompt_hypothesis);
    let user_prompt = Message::new(llm_msg::Role::User, &state.user_prompt_rendered);
    let messages = vec![system_prompt, user_prompt];
    let tools_enabled = false;

    info!("Generating hypothesis.");
    let llm_response = llm_generate::generate(messages, tools_enabled, &provider)?;
    let hypothesis = llm_response.content;

    debug!("Generation succeeded. Raw hypothesis:\n{hypothesis}");
    state.hypothesis_raw = hypothesis;
    Ok(state)
}

fn parse_hypothesis(mut state: State) -> Result<State, WorkflowError> {
    info!("Parsing hypothesis from LLM response.");
    let hypothesis: Hypothesis = serde_json::from_str(&state.hypothesis_raw)?;

    info!("Hypothesis parsing succeeded.");
    debug!("{hypothesis:?}");

    state.hypothesis = hypothesis;
    Ok(state)
}

fn write_config_files(state: State) -> Result<State, WorkflowError> {
    let contents = &state.hypothesis.prompt;
    let path = &state.config.prompt_path_experiment;

    info!("Writing experimental prompt to {path}");
    fs::write(path, contents)?;
    info!("Experimental prompt write to {path} succeeded.");

    Ok(state)
}

fn commit_configs(mut state: State) -> Result<State, WorkflowError> {
    if state.config.dry_run {
        info!("Dry-run: skipping git commit.");
        return Ok(state);
    }

    let description = &state.hypothesis.description;
    let prompt_variation = &state.config.prompt_path_experiment;

    info!("Staging {prompt_variation} changes.");
    let staging_output = Command::new("git")
        .arg("add")
        .arg(prompt_variation)
        .output()?
        .stdout_str();
    info!("{staging_output}");

    let commit_message = format!("experiment: {description}");

    info!("Committing changes with message: {commit_message}");
    let commit_output = Command::new("git")
        .arg("commit")
        .arg("-m")
        .arg(&commit_message)
        .output()?
        .stdout_str();
    info!("{commit_output}");

    info!("Retrieving experiment git hash");
    let git_hash = Command::new("git")
        .arg("rev-parse")
        .arg("--short")
        .arg("HEAD")
        .output()?
        .stdout_str();
    info!("Current experiment git hash is {git_hash}");

    state.current_experiment_hash = git_hash;
    Ok(state)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0; // or panic in debug
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot as f64
}

/// This function is directly tailored to a single experiment, and will likely benefit from being provided as a
/// dependency after the prototype implementation.
fn run_experiment(mut state: State) -> Result<State, WorkflowError> {
    let mut valid_output_count = 0;
    let mut total_relevance = 0.0;
    let mut total_queries = 0usize;

    // let path = "topics.json";
    let path = "topics.json";
    info!("Reading topics from {path}");
    let content = fs::read_to_string(path)?;
    debug!("Content from {path}:\n{content}");

    info!("Parsing topics");
    let topics: Vec<String> = serde_json::from_str(&content)?;
    debug!("Topics: {topics:?}");

    let start = Instant::now();

    for topic in &topics {
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > state.config.max_experiment_seconds {
            info!("Maximum experiment seconds exceeded: {elapsed}");
            break;
        }

        info!("Generating response for prompt variation. topic: {topic}");
        let system_prompt_variation = Message::new(Role::System, &state.sys_prompt_variation);
        let user_prompt = Message::new(Role::User, topic);

        let messages: Vec<Message> = vec![system_prompt_variation, user_prompt];
        let tools_enabled = false;
        let response_provider = llm_generate::Provider::Ollama {
            model: state.config.experiment_model.clone(),
        };
        let embedding_provider = llm_embed::Provider::Ollama {
            model: state.config.embedding_model.clone(),
        };

        let response = llm_generate::generate(messages, tools_enabled, &response_provider)?;
        debug!("Received response for topic: {topic}\n{response:?}");

        let generated_queries: Vec<String> = match serde_json::from_str(&response.content) {
            Ok(vec) => vec,
            Err(e) => {
                error!("Error parsing generated queries for {topic}.\n{e}");
                continue;
            }
        };

        debug!("Generated queries: {generated_queries:?}");

        // TODO this needs to be an experiment parameter
        let target_length = 5;
        let num_queries = generated_queries.len();

        info!("Validating queries for topic: {topic}");
        if num_queries == target_length {
            info!("Queries validated. Scoring relevance.");
            valid_output_count += 1;

            // TODO batch the embedding calls for topics and queries, and zip the cosine
            // similarity?
            debug!("Generating embedding for {topic}");
            let topic_embedding =
                llm_embed::generate(vec![topic.to_string()], &embedding_provider)?;
            debug!("Topic embedding success for {topic}");

            for query in generated_queries {
                debug!("Generating embedding for {}", query.clone());
                let query_embedding = llm_embed::generate(vec![query], &embedding_provider)?;
                debug!("Query embedding success");
                let cosine_similarity = cosine_sim(
                    &topic_embedding.embeddings[0],
                    &query_embedding.embeddings[0],
                );

                total_relevance += cosine_similarity;
                total_queries += 1;
            }
        } else {
            info!("Expected {target_length} queries but received {num_queries}.");
        }
    }

    let total_seconds = start.elapsed().as_secs_f64();
    let validity_rate = valid_output_count as f64 / topics.len() as f64;
    let avg_relevance = if total_queries > 0 {
        total_relevance / total_queries as f64
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

    state.experiment_current_result = result;
    Ok(state)
}

fn get_current_high_score(mut state: State) -> Result<State, WorkflowError> {
    let path = &state.config.log_path_results;
    info!("Reading results log from {}", path);

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            info!("Results log not found, starting fresh with high score 0.0");
            state.experiment_high_score = 0.0;
            return Ok(state);
        }
        Err(e) => return Err(e.into()),
    };

    debug!("Content from {path}:\n{content}");

    let mut high_score = 0.0;
    for line in content.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 {
            continue;
        }
        if fields[0] == "commit_hash" {
            continue;
        }
        info!("Parsing line: {:?}", fields);
        if let Ok(score) = fields[1].parse::<f64>() {
            info!("Parsed score: {} from commit {}", score, fields[0]);
            if score > high_score {
                high_score = score;
            }
        } else {
            info!("Failed to parse score from field: {}", fields[1]);
        }
    }

    info!("Current high score: {high_score}");
    state.experiment_high_score = high_score;
    Ok(state)
}

fn judge_results(mut state: State) -> Result<State, WorkflowError> {
    if state.experiment_high_score < state.experiment_current_result.score {
        state.experiment_action = WorkflowAction::Keep;
    } else {
        state.experiment_action = WorkflowAction::Discard;
    }

    Ok(state)
}

fn rollback_or_pass(state: State) -> Result<State, WorkflowError> {
    if state.config.dry_run {
        info!("Dry-run: skipping rollback/tag.");
        return Ok(state);
    }

    match state.experiment_action {
        WorkflowAction::Discard => {
            info!("Discard: rolling back latest commit.");
            let _ = Command::new("git")
                .arg("reset")
                .arg("--hard")
                .arg("HEAD~1")
                .output();
        }
        WorkflowAction::Keep => {
            info!("Keeping: tagging current run.");
            let tag_name = format!("best-{}", state.runs);
            let _ = Command::new("git").arg("tag").arg(&tag_name).output();
        }
    }
    Ok(state)
}

fn log_crash(state: &State, reason: &str) -> Result<(), WorkflowError> {
    let line = format!("exp{}\t0.0\tCRASH: {}\tcrash\t0.0\n", state.runs, reason);
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&state.config.log_path_results)?
        .write_all(line.as_bytes())?;
    Ok(())
}

fn reset_git_state() {
    let _ = Command::new("git")
        .arg("reset")
        .arg("--hard")
        .arg("HEAD~1")
        .output();
}

/// Logs experiment results to the TSV results log.
///
/// # TSV Format
/// | Column     | Description                                      |
/// |------------|-------------------------------------------------|
/// | commit_hash| Git commit hash (or `expN` for pre-commit crashes) |
/// | score      | Float 0.0-1.0 (0.0 for crashes)              |
/// | description| Experiment title or crash reason               |
/// | action     | `keep`/`discard` or crash reason               |
/// | exp_seconds| total time of experiment run in seconds |
///
/// # Example Logs
/// ```
/// a1b2c3d    0.750    Simplify prompt instructions    keep    38.750
/// a1b2c3d    0.000    CRASH experiment-timeout     crash    0.0
/// e5f6g7h    0.820    Remove verbose examples    discard    37.660
/// ```
fn log_results(state: State) -> Result<State, WorkflowError> {
    let commit_hash = if state.current_experiment_hash.is_empty() {
        "NO_HASH"
    } else {
        &state.current_experiment_hash
    };
    let score = state.experiment_current_result.score;
    let description = &state.hypothesis.description;
    let action = match state.experiment_action {
        WorkflowAction::Discard => "discard",
        WorkflowAction::Keep => "keep",
    };
    let exp_seconds = state.experiment_current_result.total_seconds;

    let line = format!(
        "{}\t{}\t{}\t{}\t{}\n",
        commit_hash, score, description, action, exp_seconds
    );

    info!("Appending to results log: {}", line.trim());
    let log_path = &state.config.log_path_results;
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?
        .write_all(line.as_bytes())?;

    Ok(state)
}
