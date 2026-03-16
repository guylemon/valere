#[derive(Default, Debug)]
pub struct ExperimentResult {
    pub avg_relevance: f64,
    pub score: f64,
    pub total_seconds: f64,
    pub validity_rate: f64,
}

#[derive(Default, Debug)]
pub struct ExperimentResultBuilder {
    pub total_seconds: f64,
    pub validity_rate: f64,
    pub avg_relevance: f64,
    pub score: f64,
}

impl ExperimentResultBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(self) -> ExperimentResult {
        ExperimentResult {
            total_seconds: self.total_seconds,
            validity_rate: self.validity_rate,
            avg_relevance: self.avg_relevance,
            score: self.score,
        }
    }

    pub fn total_seconds(mut self, total_seconds: f64) -> ExperimentResultBuilder {
        self.total_seconds = total_seconds;
        self
    }

    pub fn validity_rate(mut self, validity_rate: f64) -> ExperimentResultBuilder {
        self.validity_rate = validity_rate;
        self
    }

    pub fn avg_relevance(mut self, avg_relevance: f64) -> ExperimentResultBuilder {
        self.avg_relevance = avg_relevance;
        self
    }

    pub fn score(mut self, score: f64) -> ExperimentResultBuilder {
        self.score = score;
        self
    }
}
