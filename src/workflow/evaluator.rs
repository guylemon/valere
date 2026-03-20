use crate::config::Config;
use crate::error::WorkflowError;
use crate::experiment::{Experiment, ExperimentResult};
use crate::util::{LogRecordBuilder, append_log_row, get_high_score};
use crate::workflow::proposer::Hypothesis;

use log::info;

#[derive(Debug, Default)]
enum WorkflowAction {
    #[default]
    Discard,
    Keep,
}

#[derive(Debug)]
pub struct Evaluator {
    action: WorkflowAction,
    hypothesis: Hypothesis,
}

impl Evaluator {
    pub fn new(hypothesis: Hypothesis) -> Self {
        Self {
            action: WorkflowAction::Discard,
            hypothesis,
        }
    }

    pub fn run(&mut self, configuration: &Config) -> Result<(), WorkflowError> {
        let high_score = get_high_score(&configuration.workflow_history_log_path)?;
        info!("Current high score: {high_score}");
        let mut experiment = Experiment::new(configuration);

        experiment.run(configuration)?;
        self.action = if experiment.result.score > high_score {
            WorkflowAction::Keep
        } else {
            WorkflowAction::Discard
        };

        self.log_results(configuration, &experiment.result)?;

        Ok(())
    }

    /// Logs experiment results to the TSV results log.
    ///
    /// # TSV Format
    /// | Column     | Description                                      |
    /// |------------|-------------------------------------------------|
    /// | score      | Float 0.0-1.0 (0.0 for crashes)              |
    /// | description| Experiment title or crash reason               |
    /// | action     | `keep`/`discard` or crash reason               |
    /// | validity_rate      | Float 0.0-1.0 (0.0 for crashes)              |
    /// | avg_relevance      | Float 0.0-1.0 (0.0 for crashes)              |
    /// | exp_seconds| total time of experiment run in seconds |
    /// | prompt     | prompt variation used in experiment |
    fn log_results(
        &self,
        configuration: &Config,
        result: &ExperimentResult,
    ) -> Result<(), WorkflowError> {
        let action = match self.action {
            WorkflowAction::Discard => "discard",
            WorkflowAction::Keep => "keep",
        };

        let log_record = LogRecordBuilder::new()
            .score(result.score)
            .action(action)
            .description(&self.hypothesis.description)
            .validity_rate(result.validity_rate)
            .avg_relevance(result.avg_relevance)
            .exp_seconds(result.total_seconds)
            .prompt(&self.hypothesis.prompt)
            .build();

        info!("Appending to results log: {:?}", log_record);
        append_log_row(&configuration.workflow_history_log_path, log_record)
    }
}
