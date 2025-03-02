#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct OptimizationResult {
    pub num_low: usize,
    pub search_threshold: f64,
    pub high_threshold: f64,
    pub payout_threshold: f64,
    pub multiplier: f64,
    pub stake_percent: f64,
    pub attempts: usize,
    pub balance: f64,
    pub max_balance: f64,
    pub total_bets: u32,
    pub total_series: u32,
    pub winning_series: u32,
    pub profit: f64,
    pub initial_balance: f64,
    pub initial_stake: f64,
    pub bet_type: String,
}
impl Eq for OptimizationResult {}
impl OptimizationResult {
    #[allow(dead_code)]
    pub fn to_string(&self) -> String {

        fn format_number(num: f64) -> String {
            if num < 1000000.0 {
                format!("{:.2}", num)
            } else {
                let exp = num.abs().log10().floor();
                let mantissa = num / 10f64.powf(exp);
                format!("{:.2}E+{}", mantissa, exp as i32)
            }
        }

        format!(
            "Числа: {}, Поиск <= {:.2}, Ожидание >= {:.2}, Ставка >= {:.2}, Множитель={:.2}, \
            Процент={:.1}%, Попыток={}, Баланс={}, Макс.баланс={}, \
            Всего ставок={}, Выигрышных серий={}, Проигранных серий={}, Прибыль={}",
            self.num_low,
            self.search_threshold,
            self.high_threshold,
            self.payout_threshold,
            self.multiplier,
            self.stake_percent,
            self.attempts,
            format_number(self.balance),
            format_number(self.max_balance),
            self.total_bets,
            self.winning_series,
            self.total_series - self.winning_series,
            format_number(self.profit)
        )
    }
}
