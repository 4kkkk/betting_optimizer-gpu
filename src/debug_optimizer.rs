use ndarray::Array1;
use crate::optimization::{OptimizationResult, Params};
use std::time::Instant;

pub fn debug_optimize(params: &Params) -> (Vec<OptimizationResult>, std::time::Duration) {
    println!("DEBUG: Начинаем отладочную оптимизацию");
    let start = Instant::now();
    let mut results = Vec::new();

    // Создаем комбинации на лету и сразу их обрабатываем
    for multiplier_int in (params.min_multiplier * 10.0) as i32..=(params.max_multiplier * 10.0) as i32 {
        let multiplier = multiplier_int as f64 / 10.0;
        println!("DEBUG: Тестирование множителя {:.1}", multiplier);

        // Фиксируем один набор параметров для отладки
        let num_low = params.min_num_low;
        let search_threshold = 3.8;
        let high_threshold = 2.0;
        let payout_threshold = 4.5;
        let stake_percent = params.min_stake_percent;
        let attempts = params.min_attempts_count;

        // Запускаем стратегию с этими параметрами
        let (balance, max_balance, total_bets, total_series, winning_series, _) =
            debug_strategy(
                &params.numbers,
                params.stake,
                multiplier,
                params.initial_balance,
                search_threshold,
                high_threshold,
                payout_threshold,
                num_low,
                if params.bet_type == "fixed" { 0 } else { 1 },
                stake_percent,
                attempts as u32
            );

        println!("DEBUG: Множитель {:.1}: баланс={:.2}, прибыль={:.2}",
                 multiplier, balance, balance - params.initial_balance);

        // Проверяем, является ли стратегия прибыльной
        if total_series > 0 && balance > params.initial_balance {
            let result = OptimizationResult {
                num_low,
                search_threshold,
                high_threshold,
                payout_threshold,
                multiplier,
                stake_percent,
                attempts,
                balance,
                max_balance,
                total_bets,
                total_series,
                winning_series,
                profit: balance - params.initial_balance,
                initial_balance: params.initial_balance,
                bet_type: params.bet_type.clone(),
                initial_stake: params.stake,
            };

            results.push(result);
            println!("DEBUG: Добавлен результат для множителя {:.1}", multiplier);
        }
    }

    // Сортируем результаты по прибыли
    results.sort_by(|a, b| b.profit.partial_cmp(&a.profit).unwrap_or(std::cmp::Ordering::Equal));

    println!("DEBUG: Найдено {} прибыльных стратегий", results.len());
    (results, start.elapsed())
}

// Полностью автономная реализация стратегии для отладки
fn debug_strategy(
    numbers: &Array1<f64>,
    stake: f64,
    multiplier: f64,
    initial_balance: f64,
    search_threshold: f64,
    high_threshold: f64,
    payout_threshold: f64,
    num_low: usize,
    bet_type: i32,
    stake_percent: f64,
    attempts: u32
) -> (f64, f64, u32, u32, u32, u32) {
    let mut balance = initial_balance;
    let mut max_balance = initial_balance;
    let mut total_series = 0u32;
    let mut winning_series = 0u32;
    let mut total_bets = 0u32;
    let mut consecutive_losses = 0u32;
    let len = numbers.len();
    let mut i = num_low;

    println!("DEBUG: Запуск стратегии с множителем={:.1}, поиск={:.1}, ожидание={:.1}, ставка={:.1}",
             multiplier, search_threshold, high_threshold, payout_threshold);

    while i < len {
        let mut sequence_valid = true;
        for j in 0..num_low {
            if i <= j || numbers[i - j - 1] > search_threshold {
                sequence_valid = false;
                break;
            }
        }

        if sequence_valid {
            let mut search_i = i;
            while search_i < len && numbers[search_i] < high_threshold {
                search_i += 1;
            }

            if search_i < len && numbers[search_i] >= high_threshold {
                total_series += 1;
                let mut betting_attempts = 0;
                let mut current_i = search_i;

                let initial_bet = if bet_type == 0 {
                    stake
                } else {
                    balance * (stake_percent / 100.0)
                };

                let mut current_stake = initial_bet;

                while betting_attempts <= attempts - 1 && current_i < len - 1 {
                    if current_stake > balance {
                        break;
                    }

                    current_i += 1;
                    total_bets += 1;
                    balance -= current_stake;

                    if numbers[current_i] >= payout_threshold {
                        balance += current_stake * payout_threshold;
                        winning_series += 1;
                        consecutive_losses = 0;
                        max_balance = max_balance.max(balance);
                        break;
                    } else {
                        consecutive_losses += 1;
                        current_stake *= multiplier;
                        betting_attempts += 1;
                    }
                }

                if betting_attempts >= attempts {
                    consecutive_losses = 0;
                }
                i = current_i + 1;
                continue;
            }
        }
        i += 1;
    }

    println!("DEBUG: Завершение стратегии, баланс={:.2}, серий={}, выигрышей={}",
             balance, total_series, winning_series);

    (
        balance,
        max_balance,
        total_bets,
        total_series,
        winning_series,
        consecutive_losses,
    )
}