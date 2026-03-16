use std::fmt;

pub enum WorkflowError {
    Config(String),
    Io(std::io::Error),
    LanguageModelError(String),
    PromptError(String),
}

impl fmt::Debug for WorkflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkflowError::Config(msg) => write!(f, "Config error: {msg}"),
            WorkflowError::Io(e) => write!(f, "IO error: {e}"),
            WorkflowError::PromptError(e) => write!(f, "Prompt error: {e}"),
            WorkflowError::LanguageModelError(e) => write!(f, "LLM error: {e}"),
        }
    }
}

impl fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkflowError::Config(msg) => write!(f, "{}", msg),
            WorkflowError::Io(e) => write!(f, "{}", e),
            WorkflowError::PromptError(e) => write!(f, "{}", e),
            WorkflowError::LanguageModelError(e) => write!(f, "LLM error: {e}"),
        }
    }
}

impl std::error::Error for WorkflowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WorkflowError::Config(_) => None,
            WorkflowError::Io(e) => Some(e),
            WorkflowError::PromptError(_) => None,
            WorkflowError::LanguageModelError(_) => None,
        }
    }
}

impl From<std::io::Error> for WorkflowError {
    fn from(err: std::io::Error) -> Self {
        WorkflowError::Io(err)
    }
}

impl From<toml::de::Error> for WorkflowError {
    fn from(err: toml::de::Error) -> Self {
        WorkflowError::Config(err.to_string())
    }
}

impl From<llm_prompt::PromptError> for WorkflowError {
    fn from(err: llm_prompt::PromptError) -> Self {
        WorkflowError::PromptError(err.to_string())
    }
}

impl From<serde_json::Error> for WorkflowError {
    fn from(err: serde_json::Error) -> Self {
        WorkflowError::Config(err.to_string())
    }
}

impl From<csv::Error> for WorkflowError {
    fn from(err: csv::Error) -> Self {
        WorkflowError::Io(err.into())
    }
}
