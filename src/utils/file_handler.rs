use crate::optimization::OptimizationResult;
use ndarray::Array1;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead, BufReader};
use std::time::Instant;
use memmap2::MmapOptions;

const MAX_DETAILED_ROWS: usize = 500;

pub fn load_data_from_file(path: &str) -> io::Result<(Array1<f64>, Option<String>)> {
    println!("Загрузка данных из файла: {}", path);
    let start = Instant::now();
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut numbers = Vec::new();
    let mut error_msg = None;
    let mut valid_numbers = 0;
    let mut invalid_lines = Vec::new();

    // Используем memory mapping для больших файлов (больше 50 МБ)
    if file_size > 50 * 1024 * 1024 {
        println!("Используем memory mapping для большого файла: {} МБ", file_size / (1024 * 1024));

        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let mut current_line = Vec::new();
        let mut line_number = 1;

        for &byte in mmap.iter() {
            if byte == b'\n' {
                if let Ok(line) = String::from_utf8(current_line.clone()) {
                    let first_value = line.split_whitespace().next().unwrap_or("");
                    match first_value.parse::<f64>() {
                        Ok(num) => {
                            if num >= 0.0 {
                                numbers.push(num);
                                valid_numbers += 1;
                            } else {
                                invalid_lines.push(format!("Строка {}: отрицательное число", line_number));
                            }
                        }
                        Err(_) => {
                            if !line.trim().is_empty() {
                                invalid_lines.push(format!("Строка {}: не удалось распознать число", line_number));
                            }
                        }
                    }
                }
                current_line.clear();
                line_number += 1;
            } else {
                current_line.push(byte);
            }
        }
    } else {
        // Для небольших файлов используем обычное построчное чтение
        let reader = BufReader::with_capacity(256 * 1024, file); // Увеличенный буфер для чтения

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
                    if !first_value.is_empty() {
                        invalid_lines.push(format!("Строка {}: не удалось распознать число", i + 1));
                    }
                }
            }
        }
    }

    if !invalid_lines.is_empty() {
        let total_errors = invalid_lines.len();
        // Ограничиваем количество отображаемых ошибок
        let errors_to_show = invalid_lines.iter().take(20).cloned().collect::<Vec<_>>();
        error_msg = Some(format!(
            "Найдено {} ошибок при загрузке данных{}:\n{}",
            total_errors,
            if total_errors > 20 { format!(" (показаны первые 20 из {})", total_errors) } else { String::new() },
            errors_to_show.join("\n")
        ));
    }

    if valid_numbers == 0 {
        error_msg = Some("Файл не содержит валидных чисел".to_string());
    }

    let elapsed = start.elapsed();
    println!("Загружено {} чисел за {:?}", valid_numbers, elapsed);

    Ok((Array1::from(numbers), error_msg))
}

pub fn save_detailed_calculation(
    numbers: &Array1<f64>,
    best_result: &OptimizationResult,
    filename: &str,
) -> io::Result<()> {
    let start = Instant::now();
    let mut file = File::create(filename)?;
    let mut row_count = 0;

    // Оптимизация: предварительно выделяем строки для записи
    let mut output = String::with_capacity(1024 * 1024); // Выделяем 1 МБ буфера

    output.push_str("=== Детальный расчет для лучшей стратегии ===\n\n");
    output.push_str("Параметры стратегии:\n");
    output.push_str(&format!("Базовая ставка: {:.2}\n", best_result.initial_stake));
    output.push_str(&format!(
        "Тип ставки: {}\n",
        if best_result.bet_type == "fixed" {
            "Фиксированная"
        } else {
            "Процент от баланса"
        }
    ));
    output.push_str(&format!("Количество чисел для поиска: {}\n", best_result.num_low));
    output.push_str(&format!("Порог поиска: {:.2}\n", best_result.search_threshold));
    output.push_str(&format!("Порог ожидания: {:.2}\n", best_result.high_threshold));
    output.push_str(&format!("Порог ставки: {:.2}\n", best_result.payout_threshold));
    output.push_str(&format!("Множитель: {:.2}\n", best_result.multiplier));
    output.push_str(&format!("Процент ставки: {:.2}%\n", best_result.stake_percent));
    output.push_str(&format!("Количество попыток: {}\n\n", best_result.attempts));

    output.push_str("Пошаговый расчет:\n\n");
    output.push_str(&format!(
        "{:<5} {:<10} {:<10} {:<10} {:<15} {:<15} {:<10}\n",
        "i", "Число", "Ставка", "Выигрыш", "Баланс", "Макс.баланс", "Действие"
    ));
    output.push_str(&format!("{}\n", "-".repeat(75)));

    let mut balance = best_result.initial_balance;
    let mut max_balance = balance;
    let mut i = best_result.num_low;

    // Структура для ускорения формирования таблицы
    struct RowData {
        index: usize,
        number: f64,
        stake: String,
        win: String,
        balance: f64,
        max_balance: f64,
        action: &'static str,
    }

    let mut rows = Vec::with_capacity(MAX_DETAILED_ROWS);

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
                rows.push(RowData {
                    index: i,
                    number: numbers[i],
                    stake: "-".to_string(),
                    win: "-".to_string(),
                    balance,
                    max_balance,
                    action: "Поиск",
                });
                row_count += 1;
            } else if row_count == MAX_DETAILED_ROWS {
                output.push_str("\n... и еще строки ...\n\n");
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
                        action: "Ожидание",
                    });
                    row_count += 1;
                } else if row_count == MAX_DETAILED_ROWS {
                    output.push_str("\n... и еще строки ...\n\n");
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

                for attempt in 0..=best_result.attempts-1 {
                    if current_stake > balance {
                        break;
                    }

                    balance -= current_stake;

                    if numbers[search_i + attempt + 1] >= best_result.payout_threshold {
                        let win = current_stake * best_result.payout_threshold;
                        balance += win;
                        max_balance = max_balance.max(balance);

                        if row_count < MAX_DETAILED_ROWS {
                            rows.push(RowData {
                                index: search_i + attempt + 1,
                                number: numbers[search_i + attempt + 1],
                                stake: format!("{:.2}", current_stake),
                                win: format!("{:.2}", win),
                                balance,
                                max_balance,
                                action: "Выигрыш",
                            });
                            row_count += 1;
                        } else if row_count == MAX_DETAILED_ROWS {
                            output.push_str("\n... и еще строки ...\n\n");
                            row_count += 1;
                        }
                        break;
                    } else {
                        if row_count < MAX_DETAILED_ROWS {
                            rows.push(RowData {
                                index: search_i + attempt + 1,
                                number: numbers[search_i + attempt + 1],
                                stake: format!("{:.2}", current_stake),
                                win: "-".to_string(),
                                balance,
                                max_balance,
                                action: "Проигрыш",
                            });
                            row_count += 1;
                        } else if row_count == MAX_DETAILED_ROWS {
                            output.push_str("\n... и еще строки ...\n\n");
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

    // Форматирование и запись строк таблицы
    for row in rows {
        output.push_str(&format!(
            "{:<5} {:<10.2} {:<10} {:<10} {:<15.2} {:<15.2} {:<10}\n",
            row.index, row.number, row.stake, row.win, row.balance, row.max_balance, row.action
        ));
    }

    output.push_str("\nИтоговые результаты:\n");
    output.push_str(&format!("Начальный баланс: {:.2}\n", best_result.initial_balance));
    output.push_str(&format!("Конечный баланс: {:.2}\n", best_result.balance));
    output.push_str(&format!("Максимальный баланс: {:.2}\n", best_result.max_balance));
    output.push_str(&format!("Общая прибыль: {:.2}\n", best_result.profit));
    output.push_str(&format!("Всего ставок: {}\n", best_result.total_bets));
    output.push_str(&format!("Всего серий: {}\n", best_result.total_series));
    output.push_str(&format!("Выигрышных серий: {}\n", best_result.winning_series));

    // Записываем весь буфер за один раз, что гораздо эффективнее множества записей
    file.write_all(output.as_bytes())?;

    let elapsed = start.elapsed();
    println!("Расчет сохранен в файл за {:?}", elapsed);

    Ok(())
}