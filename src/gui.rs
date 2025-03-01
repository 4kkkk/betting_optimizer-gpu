use crate::models::settings::Settings;
use crate::optimization::{OptimizationResult, Params};
use crate::utils::file_handler::{load_data_from_file, save_detailed_calculation};
use crate::utils::settings::{load_settings, save_settings};
use chrono::Local;
use eframe::egui;
use rfd::FileDialog;
use std::fs::{self, File};
use std::io::{self, Write};
use std::time::Instant;
use std::thread;
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};



// Перечисление для передачи сообщений между потоками
enum OptimizationMessage {
    Progress(f32),
    LoadStats(std::time::Duration),
    Complete(Vec<OptimizationResult>, Option<std::time::Duration>, Option<std::time::Duration>, usize, f64),
    Error(String),
}

pub struct OptimizationApp {
    settings: Settings,
    status: String,
    is_running: bool,
    results: Vec<OptimizationResult>,
    start_time: Option<Instant>,
    progress: f32,
    performance_stats: PerformanceStats,
    message_sender: Option<Sender<OptimizationMessage>>,
    message_receiver: Option<mpsc::Receiver<OptimizationMessage>>,
    cancel_flag: Arc<AtomicBool>,
}

// Структура для отслеживания производительности
#[derive(Default)]
struct PerformanceStats {
    cpu_time: Option<std::time::Duration>,
    gpu_time: Option<std::time::Duration>,
    load_time: Option<std::time::Duration>,
    combinations: usize,
    processed_per_second: f64,
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
            progress: 0.0,
            performance_stats: PerformanceStats::default(),
            message_sender: None,
            message_receiver: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    //noinspection ALL
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
            // Добавьте проверку и очистку
            println!("Сохранение {} результатов от текущего запуска ({})",
                     self.results.len(),
                     Local::now().format("%Y-%m-%d %H:%M:%S"));

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

        // Добавляем информацию о производительности
        if let Some(load_time) = self.performance_stats.load_time {
            writeln!(file, "Время загрузки данных: {:?}", load_time)?;
        }
        if let Some(cpu_time) = self.performance_stats.cpu_time {
            writeln!(file, "Время CPU: {:?}", cpu_time)?;
        }
        if let Some(gpu_time) = self.performance_stats.gpu_time {
            writeln!(file, "Время GPU: {:?}", gpu_time)?;
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
            // Добавить после вывода других диапазонов
            writeln!(
                file,
                "Диапазон порога поиска: {} - {}",
                self.settings.min_search_threshold, self.settings.max_search_threshold
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

        // Добавляем информацию о настройках оптимизации
        if self.settings.use_gpu == "true" {
            writeln!(file, "Использование GPU: Да")?;
            writeln!(file, "Размер блока CUDA: {}", self.settings.block_size)?;
            writeln!(file, "Размер сетки CUDA: {}", self.settings.grid_size)?;
        } else {
            writeln!(file, "Использование GPU: Нет")?;
        }
        writeln!(file, "Количество потоков CPU: {}", self.settings.cpu_threads)?;

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
    //noinspection ALL
    fn run_optimization(&mut self) {
        if let Err(e) = save_settings(&self.settings, "settings.json") {
            self.status = format!("Ошибка сохранения настроек: {}", e);
            return;
        }

        if self.is_running {
            return;
        }

        self.is_running = true;
        self.cancel_flag.store(false, Ordering::SeqCst);
        self.start_time = Some(Instant::now());
        self.status = "Оптимизация запущена...".to_string();
        self.results.clear();
        self.progress = 0.0;
        self.performance_stats = PerformanceStats::default();

        // 1. Клонируем все необходимые данные до создания потока
        let settings_clone = self.settings.clone();
        let cancel_flag_clone = Arc::clone(&self.cancel_flag);

        // 2. Создаем каналы для обмена сообщениями
        let (sender, receiver) = mpsc::channel();
        self.message_receiver = Some(receiver);
        let sender_clone = sender.clone();

        thread::spawn(move || {
            // Загрузка данных с замером времени
            let load_start = Instant::now();
            let data_result = load_data_from_file(&settings_clone.file_path);
            let load_time = load_start.elapsed();

            match data_result {
                Ok((numbers, error)) => {
                    if let Some(err) = error {
                        sender_clone.send(OptimizationMessage::Error(err)).unwrap();
                        return;
                    }

                    // 3. Используем клонированные данные для создания параметров
                    let params = Params {
                        stake: settings_clone.stake.parse().unwrap_or(1.0),
                        min_multiplier: settings_clone.min_multiplier.parse().unwrap_or(1.67),
                        max_multiplier: settings_clone.max_multiplier.parse().unwrap_or(1.67),
                        initial_balance: settings_clone.initial_balance.parse().unwrap_or(500.0),
                        min_num_low: settings_clone.min_num_low.parse().unwrap_or(2),
                        max_num_low: settings_clone.max_num_low.parse().unwrap_or(5),
                        min_payout_threshold: settings_clone.min_payout_threshold.parse().unwrap_or(2.5),
                        max_payout_threshold: settings_clone.max_payout_threshold.parse().unwrap_or(3.0),
                        bet_type: settings_clone.bet_type.clone(),
                        min_stake_percent: settings_clone.min_stake_percent.parse().unwrap_or(1.0),
                        max_stake_percent: settings_clone.max_stake_percent.parse().unwrap_or(1.0),
                        min_high_threshold: settings_clone.min_high_threshold.parse().unwrap_or(2.0),
                        max_high_threshold: settings_clone.max_high_threshold.parse().unwrap_or(5.0),
                        min_search_threshold: settings_clone.min_search_threshold.parse().unwrap_or(1.5),
                        max_search_threshold: settings_clone.max_search_threshold.parse().unwrap_or(5.0),
                        min_attempts_count: settings_clone.min_attempts_count.parse().unwrap_or(4),
                        max_attempts_count: settings_clone.max_attempts_count.parse().unwrap_or(4),
                        numbers,
                        max_results: settings_clone.max_results.clone(),
                        // 4. Парсим число потоков и use_gpu из клонированных настроек
                        cpu_threads: settings_clone.cpu_threads.parse().unwrap_or(num_cpus::get()),
                        use_gpu: settings_clone.use_gpu == "true",
                    };

                    sender_clone.send(OptimizationMessage::LoadStats(load_time)).unwrap();

                    // Создаем функцию обратного вызова для отслеживания прогресса
                    let progress_sender = sender_clone.clone();
                    let cancel_flag_for_callback = Arc::clone(&cancel_flag_clone);

                    let (results, cpu_time, gpu_time) = crate::optimization::strategy::optimize_parameters(
                        &params.numbers,
                        &params,
                        move |current, total| {
                            if cancel_flag_for_callback.load(Ordering::SeqCst) {
                                return;
                            }
                            let progress = if total > 0 { current as f32 / total as f32 } else { 0.0 };
                            progress_sender.send(OptimizationMessage::Progress(progress)).unwrap_or_default();
                        },
                        &cancel_flag_clone
                    );

                    // Вычисляем статистику производительности
                    let combinations = if let Some(_) = results.first() {
                        // Примерная оценка числа комбинаций
                        (params.max_num_low - params.min_num_low + 1) *
                            ((params.max_search_threshold - params.min_search_threshold) / 0.1) as usize *
                            ((params.max_multiplier - params.min_multiplier) / 0.1) as usize
                    } else {
                        0
                    };

                    let per_second = if let Some(total_time) = match (cpu_time, gpu_time) {
                        (Some(cpu), Some(gpu)) => Some(cpu.max(gpu)),
                        (Some(cpu), None) => Some(cpu),
                        (None, Some(gpu)) => Some(gpu),
                        _ => None,
                    } {
                        if total_time.as_secs_f64() > 0.0 {
                            combinations as f64 / total_time.as_secs_f64()
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    // Если флаг отмены не установлен, отправляем результаты
                    if !cancel_flag_clone.load(Ordering::SeqCst) {
                        sender_clone.send(OptimizationMessage::Complete(
                            results, cpu_time, gpu_time, combinations, per_second
                        )).unwrap();
                    }
                }
                Err(e) => {
                    sender_clone.send(OptimizationMessage::Error(e.to_string())).unwrap();
                }
            }
        });

        // Сохраняем отправителя для возможных обновлений из UI
        self.message_sender = Some(sender);
    }


    fn display_status(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if self.is_running {
                ui.spinner();
            }
            ui.label(&self.status);
        });

        // Отображаем прогресс-бар
        if self.is_running {
            ui.add(egui::ProgressBar::new(self.progress).show_percentage());
        }

        // Отображаем статистику производительности, если доступна
        if self.performance_stats.combinations > 0 {
            ui.group(|ui| {
                ui.heading("Статистика производительности");
                if let Some(load_time) = self.performance_stats.load_time {
                    ui.label(format!("Время загрузки данных: {:?}", load_time));
                }
                if let Some(cpu_time) = self.performance_stats.cpu_time {
                    ui.label(format!("Время расчёта на CPU: {:?}", cpu_time));
                }
                if let Some(gpu_time) = self.performance_stats.gpu_time {
                    ui.label(format!("Время расчёта на GPU: {:?}", gpu_time));
                }
                ui.label(format!("Обработано комбинаций: {}", self.performance_stats.combinations));
                ui.label(format!("Скорость: {:.1} комб/с", self.performance_stats.processed_per_second));
            });
        }

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
                Self::format_number(self.results[0].balance)
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
    //noinspection ALL
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Отрисовка основного пользовательского интерфейса
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

            // Добавляем новую группу настроек для оптимизации GPU
            ui.group(|ui| {
                ui.heading("Настройки оптимизации");
                ui.horizontal(|ui| {
                    ui.label("Использовать GPU:");
                    ui.radio_value(
                        &mut self.settings.use_gpu,
                        "true".to_string(),
                        "Да",
                    );
                    ui.radio_value(
                        &mut self.settings.use_gpu,
                        "false".to_string(),
                        "Нет",
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Размер блока CUDA:");
                    ui.text_edit_singleline(&mut self.settings.block_size);
                });

                ui.horizontal(|ui| {
                    ui.label("Размер сетки CUDA:");
                    ui.text_edit_singleline(&mut self.settings.grid_size);
                });

                ui.horizontal(|ui| {
                    ui.label("Потоки CPU:");
                    ui.text_edit_singleline(&mut self.settings.cpu_threads);
                });
            });

            ui.group(|ui| {
                ui.heading("Настройки вывода");
                ui.horizontal(|ui| {
                    ui.label("Максимальное количество результатов:");
                    ui.text_edit_singleline(&mut self.settings.max_results);
                });
            });

            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Запустить оптимизацию").clicked() {
                    self.run_optimization();
                }

                if ui.button("Сохранить настройки").clicked() {
                    if let Err(e) = save_settings(&self.settings, "settings.json") {
                        self.status = format!("Ошибка сохранения настроек: {}", e);
                    } else {
                        self.status = "Настройки сохранены".to_string();
                    }
                }

                if self.is_running && ui.button("Прервать").clicked() {
                    // Устанавливаем флаг отмены
                    self.cancel_flag.store(true, Ordering::SeqCst);
                    self.is_running = false;
                    self.status = "Оптимизация прервана".to_string();
                }


            });

            ui.separator();

            if !self.results.is_empty() {
                ui.heading("Результаты оптимизации:");
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        for (i, result) in self.results.iter().take(10).enumerate() {
                            ui.group(|ui| {
                                ui.label(format!("Комбинация #{}", i + 1));
                                ui.label(format!(
                                    "Баланс: {} (макс: {})",
                                    Self::format_number(result.balance),
                                    Self::format_number(result.max_balance)
                                ));
                                ui.label(format!("Прибыль: {}", Self::format_number(result.profit)));
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

        // Проверяем сообщения от фонового потока
        if self.is_running {
            if let Some(receiver) = &self.message_receiver {
                match receiver.try_recv() {
                    Ok(message) => match message {
                        OptimizationMessage::Progress(progress) => {
                          //  println!("Получен прогресс: {:.1}%", progress * 100.0);
                            self.progress = progress;
                        },
                        OptimizationMessage::LoadStats(duration) => {
                            self.performance_stats.load_time = Some(duration);
                        },
                        OptimizationMessage::Complete(results, cpu_time, gpu_time, combinations, per_second) => {
                            self.results = results;
                            self.performance_stats.cpu_time = cpu_time;
                            self.performance_stats.gpu_time = gpu_time;
                            self.performance_stats.combinations = combinations;
                            self.performance_stats.processed_per_second = per_second;
                            self.is_running = false;
                            self.status = format!("Оптимизация завершена! Найдено {} прибыльных комбинаций.", self.results.len());

                            // Сохраняем результаты
                            if let Err(e) = self.save_results() {
                                self.status = format!("Ошибка сохранения результатов: {}", e);
                            }
                        },
                        OptimizationMessage::Error(error_msg) => {
                            self.status = error_msg;
                            self.is_running = false;
                        }
                    },
                    Err(mpsc::TryRecvError::Empty) => {
                        // Нет новых сообщений, это нормально
                    },
                    Err(mpsc::TryRecvError::Disconnected) => {
                        // Отправитель отключен, завершаем расчет
                        println!("Канал сообщений отключен");
                        self.is_running = false;
                        if self.results.is_empty() {
                            self.status = "Оптимизация прервана или завершена с ошибкой".to_string();
                        }
                    }
                }
            }

            // Запрашиваем обновление UI для проверки сообщений
            ctx.request_repaint();
        }
    }
}

