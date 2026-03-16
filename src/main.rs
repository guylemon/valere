mod config;
mod error;
mod experiment;
mod util;
mod workflow;

use log::{error, info};
use std::process;

use crate::config::Config;
use crate::workflow::Workflow;

fn main() {
    env_logger::init();
    info!("Valere started.");

    let configuration: Config = match Config::load() {
        Ok(cfg) => cfg,
        Err(e) => {
            let msg = format!("Workflow configuration failed: {e}");
            error!("{msg}");
            process::exit(1)
        }
    };
    info!("Configuration loaded.");

    if configuration.validate_only {
        info!("Validation complete.");
        process::exit(0);
    }

    if let Err(e) = Workflow::run(&configuration) {
        error!("Unrecoverable error: {e}");
        process::exit(1)
    }
}
