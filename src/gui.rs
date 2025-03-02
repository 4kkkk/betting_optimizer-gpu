use crate::models::settings::Settings;
use crate::optimization::{optimize_parameters, OptimizationResult, Params};
use crate::utils::file_handler::{load_data_from_file, save_detailed_calculation};
use crate::utils::settings::{load_settings, save_settings};
use chrono::Local;
use eframe::egui;
use rfd::FileDialog;
use std::fs::{self, File};
use std::io::{self, Write};
use std::time::Instant;

pub struct OptimizationApp {
    settings: Settings,
    status: String,
    is_running: bool,
    results: Vec<OptimizationResult>,
    start_time: Option<Instant>,
}

impl OptimizationApp {
    pub fn new() -> Self {
        let settings = load_settings("settings.json").unwrap_or_default();
        Self {
            settings,
            status: String::new(),
            is_running: false,
            results: Vec::new(),
            start_time: None,
        }
    }
    fn format_number(num: f64) -> String {
        if num < 10000.0 {
            format!("{:.2}", num)
        } else {
            let exp = num.abs().log10().floor();
            let mantissa = num / 10f64.powf(exp);
            format!("{:.2}E+{}", mantissa, exp as i32)
        }
    }

    fn select_file(&mut self) {
        if let Some(path) = FileDialog::new()
            .add_filter("Text files", &["txt"])
            .pick_file()
        {
            self.settings.file_path = path.display().to_string();
            save_settings(&self.settings, "settings.json").ok();
        }
    }

    fn save_results(&self) -> io::Result<()> {
        fs::create_dir_all("results")?;
        let input_filename = std::path::Path::new(&self.settings.file_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("default")
            .trim_end_matches(".txt");

        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let base_filename = format!("{}_{}", input_filename, timestamp);
        let results_filename = format!("results/{}_results.txt", base_filename);
        let raschet_filename = format!("results/{}_raschet.txt", base_filename);

        let mut file = File::create(&results_filename)?;

        writeln!(file, "=== ПАРАМЕТРЫ ЗАПУСКА ===\n")?;
        writeln!(
            file,
            "Время запуска: {}",
            Local::now().format("%Y-%m-%d %H:%M:%S")
        )?;

        if let Some(start) = self.start_time {
            writeln!(file, "Время выполнения: {:?}", start.elapsed())?;
        }
        writeln!(
            file,
            "Фиксированная базовая ставка: {}",
            self.settings.stake
        )?;
        writeln!(
            file,
            "Тип ставки: {}",
            if self.settings.bet_type == "fixed" {
                "Фиксированная"
            } else {
                "Процент от баланса"
            }
        )?;
        writeln!(
            file,
            "Диапазон процента от баланса: {}% - {}%",
            self.settings.min_stake_percent, self.settings.max_stake_percent
        )?;
        writeln!(
            file,
            "Диапазон множителя ставки: {} - {}",
            self.settings.min_multiplier, self.settings.max_multiplier
        )?;
        writeln!(file, "Начальный баланс: {}", self.settings.initial_balance)?;
        writeln!(
            file,
            "Диапазон чисел для поиска: {} - {}",
            self.settings.min_num_low, self.settings.max_num_low
        )?;
        writeln!(
            file,
            "Диапазон порога для ставки: {} - {}",
            self.settings.min_payout_threshold, self.settings.max_payout_threshold
        )?;
        writeln!(
            file,
            "Диапазон порога ожидания: {} - {}",
            self.settings.min_high_threshold, self.settings.max_high_threshold
        )?;
        writeln!(
            file,
            "Диапазон попыток для ставок: {} - {}",
            self.settings.min_attempts_count, self.settings.max_attempts_count
        )?;

        writeln!(file, "\n=== ЛУЧШИЕ РЕЗУЛЬТАТЫ ===\n")?;

        for result in self.results.iter() {
            writeln!(

                file,
                "Числа: {}, Поиск <= {:.2}, Ожидание >= {:.2}, Ставка >= {:.2}, Множитель={:.2}, Процент={:.1}%, Попыток={}, Баланс={}, Макс.баланс={}, Всего ставок={}, Выигрышных серий={}, Проигранных серий={}, Прибыль={}",
                result.num_low,
                result.search_threshold,
                result.high_threshold,
                result.payout_threshold,
                result.multiplier,
                result.stake_percent,
                result.attempts,
                Self::format_number(result.balance),
                Self::format_number(result.max_balance),
                result.total_bets,
                result.winning_series,
                result.total_series - result.winning_series,
                Self::format_number(result.profit)
            )?;
        }

        if !self.results.is_empty() {
            if let Ok((numbers, _)) = load_data_from_file(&self.settings.file_path) {
                save_detailed_calculation(&numbers, &self.results[0], &raschet_filename)?;
            }
        }


        Ok(())
    }

    fn run_optimization(&mut self) {
        if let Err(e) = save_settings(&self.settings, "settings.json") {
            self.status = format!("Ошибка сохранения настроек: {}", e);
            return;
        }

        if self.is_running {
            return;
        }

        self.is_running = true;
        self.start_time = Some(Instant::now()); // Устанавливаем время начала
        self.status = "Оптимизация запущена...".to_string();
        self.results.clear();

        if let Ok((numbers, error)) = load_data_from_file(&self.settings.file_path) {
            if let Some(err) = error {
                self.status = err;
                self.is_running = false;
                return;
            }

            let params = Params {
                stake: self.settings.stake.parse().unwrap_or(1.0),
                min_multiplier: self.settings.min_multiplier.parse().unwrap_or(1.67),
                max_multiplier: self.settings.max_multiplier.parse().unwrap_or(1.67),
                initial_balance: self.settings.initial_balance.parse().unwrap_or(500.0),
                min_num_low: self.settings.min_num_low.parse().unwrap_or(2),
                max_num_low: self.settings.max_num_low.parse().unwrap_or(5),
                min_payout_threshold: self.settings.min_payout_threshold.parse().unwrap_or(2.5),
                max_payout_threshold: self.settings.max_payout_threshold.parse().unwrap_or(3.0),
                bet_type: self.settings.bet_type.clone(),
                min_stake_percent: self.settings.min_stake_percent.parse().unwrap_or(1.0),
                max_stake_percent: self.settings.max_stake_percent.parse().unwrap_or(1.0),
                min_high_threshold: self.settings.min_high_threshold.parse().unwrap_or(2.0),
                max_high_threshold: self.settings.max_high_threshold.parse().unwrap_or(5.0),
                min_search_threshold: self.settings.min_search_threshold.parse().unwrap_or(1.5),
                max_search_threshold: self.settings.max_search_threshold.parse().unwrap_or(5.0),
                min_attempts_count: self.settings.min_attempts_count.parse().unwrap_or(4),
                max_attempts_count: self.settings.max_attempts_count.parse().unwrap_or(4),
                numbers,
                max_results: self.settings.max_results.clone(),
                max_combination_batch: "1000000".to_string(), // Значение по умолчанию
                search_mode: "chunked".to_string(),           // Значение по умолчанию
            };


            match optimize_parameters(&params.numbers, &params) {
                Ok(results) => {
                    self.results = results;

                    if let Err(e) = self.save_results() {
                        self.status = format!("Ошибка сохранения результатов: {}", e);
                    } else {
                        self.status = format!(
                            "Оптимизация завершена! Найдено {} прибыльных комбинаций. Результаты сохранены.",
                            self.results.len()
                        );
                    }
                },
                Err(e) => {
                    self.status = format!("Ошибка оптимизации: {}", e);
                    self.is_running = false;
                }
            }

        } else {
            self.status = "Ошибка загрузки данных".to_string();
        }

        self.is_running = false;
    }

    fn display_status(&self, ui: &mut egui::Ui) {
        fn format_number(num: f64) -> String {
            if num < 100000.0 {
                format!("{:.2}", num)
            } else {
                let exp = num.abs().log10().floor();
                let mantissa = num / 10f64.powf(exp);
                format!("{:.2}E+{}", mantissa, exp as i32)
            }
        }

        ui.horizontal(|ui| {
            if self.is_running {
                ui.spinner();
            }
            ui.label(&self.status);
        });

        if !self.results.is_empty() {
            ui.label(format!(
                "Найдено {} прибыльных комбинаций",
                self.results.len()
            ));
            ui.label(format!(
                "Лучшая прибыль: {}",
                Self::format_number(self.results[0].profit)
            ));
            ui.label(format!(
                "Лучший баланс: {}",
                format_number(self.results[0].balance)
            ));
            ui.label(format!(
                "Лучший процент выигрышей: {:.1}%",
                (self.results[0].winning_series as f64 / self.results[0].total_series as f64)
                    * 100.0
            ));
        }
    }
}

impl eframe::App for OptimizationApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Настройки оптимизации");

            ui.group(|ui| {
                ui.heading("Основные параметры");
                ui.horizontal(|ui| {
                    ui.label("Файл данных:");
                    ui.text_edit_singleline(&mut self.settings.file_path);
                    if ui.button("Обзор").clicked() {
                        self.select_file();
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Базовая ставка:");
                    ui.text_edit_singleline(&mut self.settings.stake);
                });

                ui.horizontal(|ui| {
                    ui.label("Множитель ставки:");
                    ui.label("от");
                    ui.text_edit_singleline(&mut self.settings.min_multiplier);
                    ui.label("до");
                    ui.text_edit_singleline(&mut self.settings.max_multiplier);
                });

                ui.horizontal(|ui| {
                    ui.label("Начальный баланс:");
                    ui.text_edit_singleline(&mut self.settings.initial_balance);
                });
            });

            ui.group(|ui| {
                ui.heading("Пороговые значения");
                ui.horizontal(|ui| {
                    ui.label("Числа для поиска:");
                    ui.label("от");
                    ui.text_edit_singleline(&mut self.settings.min_num_low);
                    ui.label("до");
                    ui.text_edit_singleline(&mut self.settings.max_num_low);
                });

                ui.horizontal(|ui| {
                    ui.label("Порог поиска:");
                    ui.label("от");
                    ui.text_edit_singleline(&mut self.settings.min_search_threshold);
                    ui.label("до");
                    ui.text_edit_singleline(&mut self.settings.max_search_threshold);
                });

                ui.horizontal(|ui| {
                    ui.label("Порог ожидания:");
                    ui.label("от");
                    ui.text_edit_singleline(&mut self.settings.min_high_threshold);
                    ui.label("до");
                    ui.text_edit_singleline(&mut self.settings.max_high_threshold);
                });

                ui.horizontal(|ui| {
                    ui.label("Порог ставки:");
                    ui.label("от");
                    ui.text_edit_singleline(&mut self.settings.min_payout_threshold);
                    ui.label("до");
                    ui.text_edit_singleline(&mut self.settings.max_payout_threshold);
                });
            });

            ui.group(|ui| {
                ui.heading("Настройки ставок");
                ui.horizontal(|ui| {
                    ui.label("Тип ставки:");
                    ui.radio_value(
                        &mut self.settings.bet_type,
                        "fixed".to_string(),
                        "Фиксированная",
                    );
                    ui.radio_value(
                        &mut self.settings.bet_type,
                        "percent".to_string(),
                        "Процент от баланса",
                    );
                });

                if self.settings.bet_type == "percent" {
                    ui.horizontal(|ui| {
                        ui.label("Процент ставки:");
                        ui.label("от");
                        ui.text_edit_singleline(&mut self.settings.min_stake_percent);
                        ui.label("до");
                        ui.text_edit_singleline(&mut self.settings.max_stake_percent);
                        ui.label("%");
                    });
                }

                ui.horizontal(|ui| {
                    ui.label("Количество попыток:");
                    ui.label("от");
                    ui.text_edit_singleline(&mut self.settings.min_attempts_count);
                    ui.label("до");
                    ui.text_edit_singleline(&mut self.settings.max_attempts_count);
                });
            });
            // В функции update добавим новую группу настроек
            ui.group(|ui| {
                ui.heading("Настройки вывода");
                ui.horizontal(|ui| {
                    ui.label("Максимальное количество результатов:");
                    ui.text_edit_singleline(&mut self.settings.max_results);
                });
            });

            ui.separator();

            if ui.button("Запустить оптимизацию").clicked() {
                self.run_optimization();
            }

            ui.separator();

            if !self.results.is_empty() {
                ui.heading("Результаты оптимизации:");
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        for (i, result) in self.results.iter().take(10).enumerate() {
                            ui.group(|ui| {
                                fn format_number(num: f64) -> String {
                                    if num < 10000.0 {
                                        format!("{:.2}", num)
                                    } else {
                                        let exp = num.abs().log10().floor();
                                        let mantissa = num / 10f64.powf(exp);
                                        format!("{:.2}E+{}", mantissa, exp as i32)
                                    }
                                }
                                ui.label(format!("Комбинация #{}", i + 1));
                                ui.label(format!(
                                    "Баланс: {} (макс: {})",
                                    Self::format_number(result.balance),
                                    Self::format_number(result.max_balance)
                                ));
                                ui.label(format!("Прибыль: {}", format_number(result.profit)));
                                ui.label(format!("Всего ставок: {}", result.total_bets));
                                ui.label(format!(
                                    "Серий: {} (выиграно: {}, проиграно: {})",
                                    result.total_series,
                                    result.winning_series,
                                    result.total_series - result.winning_series
                                ));
                                ui.label(format!(
                                    "Параметры: поиск <= {:.2}, ставка >= {:.2}, множитель = {:.2}",
                                    result.search_threshold,
                                    result.payout_threshold,
                                    result.multiplier
                                ));
                            });
                            ui.separator();
                        }
                    });
            }

            self.display_status(ui);
        });
    }
}
