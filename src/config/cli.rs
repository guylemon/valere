use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "valere")]
#[command(version = "0.1.0")]
#[command(about = "Run LLM-powered experiments", long_about = None)]
pub struct CliArgs {
    #[arg(long, env = "VALERE_DRY_RUN")]
    #[arg(default_value_t = false)]
    pub dry_run: bool,

    #[arg(long, env = "VALERE_VALIDATE")]
    #[arg(default_value_t = false)]
    pub validate: bool,

    #[arg(long, env = "VALERE_CONFIG")]
    pub config: Option<std::path::PathBuf>,

    #[arg(long, env = "VALERE_EMBEDDING_MODEL")]
    pub embedding_model: Option<String>,

    #[arg(long, env = "VALERE_EXPERIMENT_MODEL")]
    pub experiment_model: Option<String>,

    #[arg(long, env = "VALERE_HYPOTHESIS_MODEL")]
    pub hypothesis_model: Option<String>,

    #[arg(long, env = "VALERE_MAX_RUNS")]
    pub max_runs: Option<usize>,

    #[arg(long, env = "VALERE_MAX_SECONDS")]
    pub max_seconds: Option<f64>,
}
