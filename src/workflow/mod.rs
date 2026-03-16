mod evaluator;
mod proposer;

use log::{error, info};

use crate::config::Config;
use crate::error::WorkflowError;
use crate::workflow::evaluator::Evaluator;
use crate::workflow::proposer::Proposer;

pub struct Workflow {}

impl Workflow {
    pub fn run(configuration: &Config) -> Result<(), WorkflowError> {
        let max_runs = configuration.experiment_max_runs;
        let mut run = 1;

        info!("Commencing workflow run for {max_runs} repetitions");

        while run <= max_runs {
            info!("-----------------------------");
            info!("START RUN {run} OF {max_runs}");
            info!("-----------------------------");

            let mut proposer = Proposer::new();
            match proposer.run(configuration) {
                Ok(_) => {}
                Err(e) => match e {
                    WorkflowError::LanguageModelError(e) => {
                        error!("Retrying. Received LanguageModelError: {e}");
                        continue;
                    }
                    unrecoverable => return Err(unrecoverable),
                },
            }

            let mut evaluator = Evaluator::new(proposer.hypothesis);
            match evaluator.run(configuration) {
                Ok(_) => {}
                Err(e) => match e {
                    WorkflowError::LanguageModelError(e) => {
                        error!("Retrying. Received LanguageModelError: {e}");
                        continue;
                    }
                    unrecoverable => return Err(unrecoverable),
                },
            }

            run += 1;
        }

        Ok(())
    }
}
