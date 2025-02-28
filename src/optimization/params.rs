use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct Params {
    pub stake: f64,
    pub min_multiplier: f64,
    pub max_multiplier: f64,
    pub initial_balance: f64,
    pub min_num_low: usize,
    pub max_num_low: usize,
    pub min_payout_threshold: f64,
    pub max_payout_threshold: f64,
    pub bet_type: String,
    pub min_stake_percent: f64,
    pub max_stake_percent: f64,
    pub min_high_threshold: f64,
    pub max_high_threshold: f64,
    pub min_search_threshold: f64,
    pub max_search_threshold: f64,
    pub min_attempts_count: usize,
    pub max_attempts_count: usize,
    pub numbers: Array1<f64>,
    pub max_results: String,
    pub use_gpu: bool,
    pub cpu_threads: usize,
}
