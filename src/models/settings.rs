use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub file_path: String,
    pub stake: String,
    pub min_multiplier: String,
    pub max_multiplier: String,
    pub initial_balance: String,
    pub min_num_low: String,
    pub max_num_low: String,
    pub min_payout_threshold: String,
    pub max_payout_threshold: String,
    pub bet_type: String,
    pub min_stake_percent: String,
    pub max_stake_percent: String,
    pub min_high_threshold: String,
    pub max_high_threshold: String,
    pub min_search_threshold: String,
    pub max_search_threshold: String,
    pub min_attempts_count: String,
    pub max_attempts_count: String,
    // Добавляем новые поля с значениями по умолчанию
    pub max_balance_threshold: String,
    pub series_win_rate_threshold: String,
    pub max_results: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            file_path: "data.txt".to_string(),
            stake: "10".to_string(),
            min_multiplier: "1.67".to_string(),
            max_multiplier: "1.67".to_string(),
            initial_balance: "500".to_string(),
            min_num_low: "2".to_string(),
            max_num_low: "5".to_string(),
            min_payout_threshold: "2.5".to_string(),
            max_payout_threshold: "3".to_string(),
            bet_type: "fixed".to_string(),
            min_stake_percent: "1".to_string(),
            max_stake_percent: "2".to_string(),
            min_high_threshold: "2".to_string(),
            max_high_threshold: "5".to_string(),
            min_search_threshold: "1.5".to_string(),
            max_search_threshold: "5".to_string(),
            min_attempts_count: "4".to_string(),
            max_attempts_count: "4".to_string(),
            max_balance_threshold: "1000".to_string(),
            series_win_rate_threshold: "50".to_string(),
            max_results: "10000".to_string(),
        }
    }
}

