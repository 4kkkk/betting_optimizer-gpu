use crate::optimization::OptimizationResult;
use ndarray::Array1;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead};

const MAX_DETAILED_ROWS: usize = 100;

pub fn load_data_from_file(path: &str) -> io::Result<(Array1<f64>, Option<String>)> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut numbers = Vec::new();
    let mut error_msg = None;
    let mut valid_numbers = 0;
    let mut invalid_lines = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let first_value = line.split_whitespace().next().unwrap_or("");

        match first_value.parse::<f64>() {
            Ok(num) => {
                if num >= 0.0 {
                    numbers.push(num);
                    valid_numbers += 1;
                } else {
                    invalid_lines.push(format!("Строка {}: отрицательное число", i + 1));
                }
            }
            Err(_) => {
                invalid_lines.push(format!("Строка {}: не удалось распознать число", i + 1));
            }
        }
    }

    if !invalid_lines.is_empty() {
        error_msg = Some(format!(
            "Найдено {} ошибок при загрузке данных:\n{}",
            invalid_lines.len(),
            invalid_lines.join("\n")
        ));
    }

    if valid_numbers == 0 {
        error_msg = Some("Файл не содержит валидных чисел".to_string());
    }

    Ok((Array1::from(numbers), error_msg))
}

pub fn save_detailed_calculation(
    numbers: &Array1<f64>,
    best_result: &OptimizationResult,
    filename: &str,
) -> io::Result<()> {
    let mut file = File::create(filename)?;
    let mut row_count = 0;

    writeln!(file, "=== Детальный расчет для лучшей стратегии ===\n")?;
    writeln!(file, "Параметры стратегии:")?;
    writeln!(file, "Базовая ставка: {:.2}", best_result.initial_stake)?;
    writeln!(
        file,
        "Тип ставки: {}",
        if best_result.bet_type == "fixed" {
            "Фиксированная"
        } else {
            "Процент от баланса"
        }
    )?;
    writeln!(file, "Количество чисел для поиска: {}", best_result.num_low)?;
    writeln!(file, "Порог поиска: {:.2}", best_result.search_threshold)?;
    writeln!(file, "Порог ожидания: {:.2}", best_result.high_threshold)?;
    writeln!(file, "Порог ставки: {:.2}", best_result.payout_threshold)?;
    writeln!(file, "Множитель: {:.2}", best_result.multiplier)?;
    writeln!(file, "Процент ставки: {:.2}%", best_result.stake_percent)?;
    writeln!(file, "Количество попыток: {}\n", best_result.attempts)?;

    writeln!(file, "Пошаговый расчет:\n")?;
    writeln!(
        file,
        "{:<5} {:<10} {:<10} {:<10} {:<15} {:<15} {:<10}",
        "i", "Число", "Ставка", "Выигрыш", "Баланс", "Макс.баланс", "Действие"
    )?;
    writeln!(file, "{}", "-".repeat(75))?;

    let mut balance = best_result.initial_balance;
    let mut max_balance = balance;
    let mut i = best_result.num_low;

    while i < numbers.len() {
        let mut sequence_valid = true;

        for j in 0..best_result.num_low {
            if i <= j || numbers[i - j - 1] > best_result.search_threshold {
                sequence_valid = false;
                break;
            }
        }

        if sequence_valid {
            if row_count < MAX_DETAILED_ROWS {
                writeln!(
                    file,
                    "{:<5} {:<10.2} {:<10} {:<10} {:<15.2} {:<15.2} {:<10}",
                    i, numbers[i], "-", "-", balance, max_balance, "Поиск"
                )?;
                row_count += 1;
            } else if row_count == MAX_DETAILED_ROWS {
                writeln!(file, "\n... и еще строки ...\n")?;
                row_count += 1;
            }

            let mut search_i = i;
            while search_i < numbers.len() && numbers[search_i] < best_result.high_threshold {
                if row_count < MAX_DETAILED_ROWS {
                    writeln!(
                        file,
                        "{:<5} {:<10.2} {:<10} {:<10} {:<15.2} {:<15.2} {:<10}",
                        search_i, numbers[search_i], "-", "-", balance, max_balance, "Ожидание"
                    )?;
                    row_count += 1;
                } else if row_count == MAX_DETAILED_ROWS {
                    writeln!(file, "\n... и еще строки ...\n")?;
                    row_count += 1;
                }
                search_i += 1;
            }

            if search_i < numbers.len() && numbers[search_i] >= best_result.high_threshold {
                let mut current_stake = if best_result.bet_type == "fixed" {
                    best_result.initial_stake
                } else {
                    balance * (best_result.stake_percent / 100.0)
                };

                for attempt in 0..=best_result.attempts-1 {                    if current_stake > balance {
                        break;
                    }

                    balance -= current_stake;

                    if numbers[search_i + attempt + 1] >= best_result.payout_threshold {
                        let win = current_stake * best_result.payout_threshold;
                        balance += win;
                        max_balance = max_balance.max(balance);

                        if row_count < MAX_DETAILED_ROWS {
                            writeln!(
                                file,
                                "{:<5} {:<10.2} {:<10.2} {:<10.2} {:<15.2} {:<15.2} {:<10}",
                                search_i + attempt + 1,
                                numbers[search_i + attempt + 1],
                                current_stake,
                                win,
                                balance,
                                max_balance,
                                "Выигрыш"
                            )?;
                            row_count += 1;
                        } else if row_count == MAX_DETAILED_ROWS {
                            writeln!(file, "\n... и еще строки ...\n")?;
                            row_count += 1;
                        }
                        break;
                    } else {
                        if row_count < MAX_DETAILED_ROWS {
                            writeln!(
                                file,
                                "{:<5} {:<10.2} {:<10.2} {:<10} {:<15.2} {:<15.2} {:<10}",
                                search_i + attempt + 1,
                                numbers[search_i + attempt + 1],
                                current_stake,
                                "-",
                                balance,
                                max_balance,
                                "Проигрыш"
                            )?;
                            row_count += 1;
                        } else if row_count == MAX_DETAILED_ROWS {
                            writeln!(file, "\n... и еще строки ...\n")?;
                            row_count += 1;
                        }
                        current_stake *= best_result.multiplier;
                    }
                }
                i = search_i + best_result.attempts + 1;
                continue;
            }
        }
        i += 1;
    }

    writeln!(file, "\nИтоговые результаты:")?;
    writeln!(file, "Начальный баланс: {:.2}", best_result.initial_balance)?;
    writeln!(file, "Конечный баланс: {:.2}", best_result.balance)?;
    writeln!(file, "Максимальный баланс: {:.2}", best_result.max_balance)?;
    writeln!(file, "Общая прибыль: {:.2}", best_result.profit)?;
    writeln!(file, "Всего ставок: {}", best_result.total_bets)?;
    writeln!(file, "Всего серий: {}", best_result.total_series)?;
    writeln!(file, "Выигрышных серий: {}", best_result.winning_series)?;

    Ok(())
}
