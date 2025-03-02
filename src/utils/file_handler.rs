use crate::optimization::OptimizationResult;
use ndarray::Array1;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead};

const MAX_DETAILED_ROWS: usize = 5000;

// Функция для округления до двух знаков после запятой (до сотых - центов)
fn round_to_cents(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

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

// Структура для хранения данных строки отчета
struct RowData {
    index: usize,
    number: f64,
    stake: String,
    win: String,
    balance: f64,
    max_balance: f64,
    action: String,
    check_info: String,
}

pub fn save_detailed_calculation(
    numbers: &Array1<f64>,
    best_result: &OptimizationResult,
    filename: &str,
) -> io::Result<()> {
    let mut file = File::create(filename)?;
    let mut row_count = 0;
    let mut actual_total_bets = 0;
    let mut actual_total_series = 0;
    let mut actual_winning_series = 0;
    let mut rows = Vec::new();

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
        "{:<5} {:<10} {:<10} {:<10} {:<15} {:<15} {:<12} {:<30}",
        "i", "Число", "Ставка", "Выигрыш", "Баланс", "Макс.баланс", "Действие", "Доп.инфо"
    )?;
    writeln!(file, "{}", "-".repeat(110))?;

    let mut balance = round_to_cents(best_result.initial_balance);
    let mut max_balance = balance;
    let mut i = best_result.num_low;

    while i < numbers.len() {
        // Показываем, что проверяем условия для начала цикла
        if row_count < MAX_DETAILED_ROWS {
            rows.push(RowData {
                index: i,
                number: numbers[i],
                stake: "-".to_string(),
                win: "-".to_string(),
                balance,
                max_balance,
                action: "Проверка".to_string(),
                check_info: format!(
                    "Проверяем {} предыдущих чисел на <= {:.2}",
                    best_result.num_low, best_result.search_threshold
                ),
            });
            row_count += 1;
        }

        let mut sequence_valid = true;

        for j in 0..best_result.num_low {
            if i <= j || numbers[i - j - 1] > best_result.search_threshold {
                sequence_valid = false;

                // Показываем, почему условие не выполнено
                if row_count < MAX_DETAILED_ROWS {
                    rows.push(RowData {
                        index: i,
                        number: numbers[i],
                        stake: "-".to_string(),
                        win: "-".to_string(),
                        balance,
                        max_balance,
                        action: "НЕУДАЧА".to_string(),
                        check_info: if i <= j {
                            "Недостаточно предыдущих чисел".to_string()
                        } else {
                            format!(
                                "Число {} > порога {:.2}",
                                numbers[i - j - 1], best_result.search_threshold
                            )
                        },
                    });
                    row_count += 1;
                }
                break;
            }
        }

        if sequence_valid {
            if row_count < MAX_DETAILED_ROWS {
                rows.push(RowData {
                    index: i,
                    number: numbers[i],
                    stake: "-".to_string(),
                    win: "-".to_string(),
                    balance,
                    max_balance,
                    action: "Поиск".to_string(),
                    check_info: format!(
                        "УСПЕШНО: найдено {} чисел <= {:.2}",
                        best_result.num_low, best_result.search_threshold
                    ),
                });
                row_count += 1;
            }

            let mut search_i = i;
            while search_i < numbers.len() && numbers[search_i] < best_result.high_threshold {
                if row_count < MAX_DETAILED_ROWS {
                    rows.push(RowData {
                        index: search_i,
                        number: numbers[search_i],
                        stake: "-".to_string(),
                        win: "-".to_string(),
                        balance,
                        max_balance,
                        action: "Ожидание".to_string(),
                        check_info: format!("Ждем число >= {:.2}", best_result.high_threshold),
                    });
                    row_count += 1;
                }
                search_i += 1;
            }

            if search_i < numbers.len() && numbers[search_i] >= best_result.high_threshold {
                // Нашли подходящее высокое число - потенциально можем делать ставку
                if row_count < MAX_DETAILED_ROWS {
                    rows.push(RowData {
                        index: search_i,
                        number: numbers[search_i],
                        stake: "-".to_string(),
                        win: "-".to_string(),
                        balance,
                        max_balance,
                        action: "СИГНАЛ".to_string(),
                        check_info: format!(
                            "Найдено число {:.2} >= порога {:.2}",
                            numbers[search_i], best_result.high_threshold
                        ),
                    });
                    row_count += 1;
                }

                actual_total_series += 1;

                // Основное исправление: Упрощаем логику и делаем как в рабочем коде
                // Всегда вычисляем свежую начальную ставку на основе текущего баланса
                let mut current_stake = if best_result.bet_type == "fixed" {
                    round_to_cents(best_result.initial_stake)
                } else {
                    round_to_cents(balance * (best_result.stake_percent / 100.0))
                };

                let mut current_i = search_i;

                // Удаляем сложные проверки can_make_all_bets, которые могут сохранять состояние
                // между сериями ставок, и заменяем их простой проверкой в цикле

                for attempt in 0..best_result.attempts {
                    if current_i >= numbers.len() - 1 {
                        if row_count < MAX_DETAILED_ROWS {
                            rows.push(RowData {
                                index: current_i,
                                number: if current_i < numbers.len() { numbers[current_i] } else { 0.0 },
                                stake: "-".to_string(),
                                win: "-".to_string(),
                                balance,
                                max_balance,
                                action: "ОСТАНОВКА".to_string(),
                                check_info: "Достигнут конец данных".to_string(),
                            });
                            row_count += 1;
                        }
                        break;
                    }

                    // Простая проверка возможности сделать ставку
                    if current_stake > balance {
                        if row_count < MAX_DETAILED_ROWS {
                            rows.push(RowData {
                                index: current_i,
                                number: numbers[current_i],
                                stake: format!("{:.2}", current_stake),
                                win: "-".to_string(),
                                balance,
                                max_balance,
                                action: "СТОП".to_string(),
                                check_info: format!(
                                    "Недостаточно средств для ставки: требуется {:.2}, доступно {:.2}",
                                    current_stake, balance
                                ),
                            });
                            row_count += 1;
                        }
                        break;
                    }

                    actual_total_bets += 1;
                    current_i += 1;

                    // Обновляем баланс, вычитая текущую ставку
                    let new_balance = round_to_cents(balance - current_stake);

                    if numbers[current_i] >= best_result.payout_threshold {
                        // Расчет выигрыша на основе порога ставки
                        let win = round_to_cents(current_stake * best_result.payout_threshold);
                        // Обновляем баланс, добавляя выигрыш
                        let updated_balance = round_to_cents(new_balance + win);
                        max_balance = max_balance.max(updated_balance);
                        actual_winning_series += 1;

                        if row_count < MAX_DETAILED_ROWS {
                            rows.push(RowData {
                                index: current_i,
                                number: numbers[current_i],
                                stake: format!("{:.2}", current_stake),
                                win: format!("{:.2}", win),
                                balance: updated_balance,
                                max_balance,
                                action: "Выигрыш".to_string(),
                                check_info: format!("Коэф: {:.2} >= порога {:.2}",
                                                    numbers[current_i], best_result.payout_threshold),
                            });
                            row_count += 1;
                        }

                        balance = updated_balance;
                        break;
                    } else {
                        if row_count < MAX_DETAILED_ROWS {
                            rows.push(RowData {
                                index: current_i,
                                number: numbers[current_i],
                                stake: format!("{:.2}", current_stake),
                                win: "-".to_string(),
                                balance: new_balance,
                                max_balance,
                                action: "Проигрыш".to_string(),
                                check_info: format!("Коэф: {:.2} < порога {:.2}",
                                                    numbers[current_i], best_result.payout_threshold),
                            });
                            row_count += 1;
                        }

                        balance = new_balance;
                        // Увеличиваем ставку для следующей попытки
                        current_stake = round_to_cents(current_stake * best_result.multiplier);
                    }
                }

                i = current_i + 1;
                continue;
            }
        }
        i += 1;
    }



    // Отсортируем строки по индексу для хронологического порядка
    rows.sort_by_key(|row| row.index);

    // Запишем все строки в файл
    for row in rows {
        writeln!(
            file,
            "{:<5} {:<10.2} {:<10} {:<10} {:<15.2} {:<15.2} {:<12} {:<}",
            row.index,
            row.number,
            row.stake,
            row.win,
            row.balance,
            row.max_balance,
            row.action,
            row.check_info
        )?;
    }

    // Запишем итоговые результаты
    writeln!(file, "\nИтоговые результаты:")?;
    writeln!(file, "Начальный баланс: {:.2}", best_result.initial_balance)?;
    writeln!(file, "Конечный баланс: {:.2}", balance)?;
    writeln!(file, "Максимальный баланс: {:.2}", max_balance)?;
    writeln!(file, "Общая прибыль: {:.2}", round_to_cents(balance - best_result.initial_balance))?;
    writeln!(file, "Всего ставок: {}", actual_total_bets)?;
    writeln!(file, "Всего серий: {}", actual_total_series)?;
    writeln!(file, "Выигрышных серий: {}", actual_winning_series)?;
    writeln!(
        file,
        "Процент выигрышных серий: {:.2}%",
        if actual_total_series > 0 {
            (actual_winning_series as f64 / actual_total_series as f64) * 100.0
        } else {
            0.0
        }
    )?;

    // Сравнение с результатами оптимизации
    writeln!(file, "\nСравнение с результатами оптимизации:")?;
    writeln!(
        file,
        "Баланс: оптимизация={:.2}, расчет={:.2}, разница={:.2}",
        best_result.balance,
        balance,
        round_to_cents(balance - best_result.balance)
    )?;
    writeln!(
        file,
        "Макс. баланс: оптимизация={:.2}, расчет={:.2}, разница={:.2}",
        best_result.max_balance,
        max_balance,
        round_to_cents(max_balance - best_result.max_balance)
    )?;
    writeln!(
        file,
        "Всего ставок: оптимизация={}, расчет={}, разница={}",
        best_result.total_bets,
        actual_total_bets,
        actual_total_bets as i64 - best_result.total_bets as i64
    )?;
    writeln!(
        file,
        "Всего серий: оптимизация={}, расчет={}, разница={}",
        best_result.total_series,
        actual_total_series,
        actual_total_series as i64 - best_result.total_series as i64
    )?;
    writeln!(
        file,
        "Выигрышных серий: оптимизация={}, расчет={}, разница={}",
        best_result.winning_series,
        actual_winning_series,
        actual_winning_series as i64 - best_result.winning_series as i64
    )?;

    Ok(())
}