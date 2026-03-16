use crate::error::WorkflowError;

use csv::ReaderBuilder;
use csv::WriterBuilder;
use log::{debug, info};
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::fs::OpenOptions;

#[derive(Debug, Deserialize, Serialize)]
pub struct LogRecord {
    score: f64,
    action: String,
    description: String,
    validity_rate: f64,
    avg_relevance: f64,
    #[serde(rename = "exp_seconds")]
    exp_seconds: f64,
    prompt: String,
}

impl LogRecord {
    pub fn score(&self) -> f64 {
        self.score
    }

    // pub fn description(&self) -> &str {
    //     &self.description
    // }
    //
    // pub fn action(&self) -> &str {
    //     &self.action
    // }
    //
    // pub fn exp_seconds(&self) -> f64 {
    //     self.exp_seconds
    // }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }
}

#[derive(Default)]
pub struct LogRecordBuilder {
    score: Option<f64>,
    action: Option<String>,
    description: Option<String>,
    validity_rate: Option<f64>,
    avg_relevance: Option<f64>,
    exp_seconds: Option<f64>,
    prompt: Option<String>,
}

impl LogRecordBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn score(mut self, score: f64) -> Self { self.score = Some(score);
        self
    }

    pub fn validity_rate(mut self, validity_rate: f64) -> Self {
        self.validity_rate = Some(validity_rate);
        self
    }

    pub fn avg_relevance(mut self, avg_relevance: f64) -> Self {
        self.avg_relevance = Some(avg_relevance);
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }

    pub fn exp_seconds(mut self, exp_seconds: f64) -> Self {
        self.exp_seconds = Some(exp_seconds);
        self
    }

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn build(self) -> LogRecord {
        LogRecord {
            score: self.score.unwrap_or(0.0),
            action: self.action.unwrap_or_default(),
            description: self.description.unwrap_or_default(),
            validity_rate: self.validity_rate.unwrap_or(0.0),
            avg_relevance: self.avg_relevance.unwrap_or(0.0),
            exp_seconds: self.exp_seconds.unwrap_or(0.0),
            prompt: self.prompt.unwrap_or_default(),
        }
    }
}

pub fn append_log_row(path: &str, record: LogRecord) -> Result<(), WorkflowError> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;

    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_writer(file);

    wtr.serialize(record)?;
    wtr.flush()?;
    Ok(())
}

pub fn read_workflow_input_file(file_path: &str) -> Result<String, WorkflowError> {
    info!("Reading from {file_path}");
    let content = fs::read_to_string(file_path)?;
    debug!("Content from {file_path}:\n{content}");
    Ok(content)
}

pub fn get_high_score(path: &str) -> Result<f64, WorkflowError> {
    let file = fs::File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(file);

    let mut high_score = 0.0;
    for result in rdr.deserialize() {
        let record: LogRecord = result?;
        if record.score() > high_score {
            high_score = record.score();
        }
    }
    Ok(high_score)
}

pub fn read_history_log_first_two_columns(path: &str) -> Result<String, WorkflowError> {
    let file = fs::File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(file);

    let mut output = String::new();

    for result in rdr.records() {
        let record = result?;
        if output.is_empty() {
            output.push_str("score\tdescription\n");
        }
        if let (Some(score), Some(desc)) = (record.get(0), record.get(1)) {
            output.push_str(&format!("{score}\t{desc}\n"));
        }
    }

    Ok(output)
}
