use crate::optimization::cuda::check_cuda_availability;
use crate::optimization::{OptimizationResult, Params};
use core_affinity;
use crossbeam::queue::ArrayQueue;
use ndarray::Array1;
use rayon::prelude::*;
use rustacuda::launch;
use rustacuda::memory::DeviceBuffer;
use rustacuda::prelude::*;
use rustacuda_derive::DeviceCopy;
use std::ffi::CString;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use windows_sys::Win32::System::Threading::{
    GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST,
};

// Оптимизация для i7-13700KF (24 потока)
// RTX 3060 поддерживает до 1024 потоков в блоке
const BLOCK_SIZE: u32 = 1024;
const GRID_SIZE: u32 = 64;

// Размер батча для обработки данных
const BATCH_SIZE: usize = 5000;
#[repr(C)]
#[derive(Clone, Copy, DeviceCopy, Debug)]
struct GpuOptimizationResult {
    balance: f64,
    max_balance: f64,
    total_bets: i32,
    total_series: i32,
    winning_series: i32,
    profit: f64,
    initial_balance: f64,
}

#[derive(Clone, Debug)]
struct ParameterCombination {
    num_low: usize,
    search_threshold: f64,
    high_threshold: f64,
    payout_threshold: f64,
    multiplier: f64,
    stake_percent: f64,
    attempts: usize,
}
fn generate_parameter_combinations(params: &Params) -> Vec<ParameterCombination> {
    // Используем целые числа для шагов, чтобы избежать проблем с плавающей точкой
    const PRECISION: i32 = 10; // Множитель для преобразования чисел с плавающей точкой в целые

    let mut combinations = Vec::new();
    println!("Начинаем генерацию комбинаций параметров...");

    // Преобразуем границы диапазонов в целые числа для более точных вычислений
    let min_mult_int = (params.min_multiplier * PRECISION as f64) as i32;
    let max_mult_int = (params.max_multiplier * PRECISION as f64) as i32;
    let step_int = 1; // Шаг 0.1 при PRECISION = 10

    println!("Диапазон множителей: {:.1}-{:.1} (в целых числах: {}-{})",
             params.min_multiplier, params.max_multiplier, min_mult_int, max_mult_int);

    // Количество итераций для множителя (включая конечную точку)
    let mult_iterations = (max_mult_int - min_mult_int) / step_int + 1;
    println!("Итераций по множителю: {}", mult_iterations);

    for num_low in params.min_num_low..=params.max_num_low {
        // Аналогично преобразуем другие параметры
        let min_search_int = (params.min_search_threshold * PRECISION as f64) as i32;
        let max_search_int = (params.max_search_threshold * PRECISION as f64) as i32;
        let search_iterations = (max_search_int - min_search_int) / step_int + 1;

        let min_high_int = (params.min_high_threshold * PRECISION as f64) as i32;
        let max_high_int = (params.max_high_threshold * PRECISION as f64) as i32;
        let high_iterations = (max_high_int - min_high_int) / step_int + 1;

        let min_payout_int = (params.min_payout_threshold * PRECISION as f64) as i32;
        let max_payout_int = (params.max_payout_threshold * PRECISION as f64) as i32;
        let payout_iterations = (max_payout_int - min_payout_int) / step_int + 1;

        println!("Итераций: множитель={}, поиск={}, ожидание={}, ставка={}",
                 mult_iterations, search_iterations, high_iterations, payout_iterations);

        // Генерируем комбинации, используя целочисленные индексы
        for m_idx in 0..mult_iterations {
            let multiplier = (min_mult_int + m_idx * step_int) as f64 / PRECISION as f64;

            for s_idx in 0..search_iterations {
                let search_threshold = (min_search_int + s_idx * step_int) as f64 / PRECISION as f64;

                for h_idx in 0..high_iterations {
                    let high_threshold = (min_high_int + h_idx * step_int) as f64 / PRECISION as f64;

                    for p_idx in 0..payout_iterations {
                        let payout_threshold = (min_payout_int + p_idx * step_int) as f64 / PRECISION as f64;

                        // Определяем stake_percent в зависимости от типа ставки
                        let stake_percent_range = if params.bet_type == "fixed" {
                            vec![params.min_stake_percent]
                        } else {
                            let min_stake_int = (params.min_stake_percent * PRECISION as f64) as i32;
                            let max_stake_int = (params.max_stake_percent * PRECISION as f64) as i32;
                            let stake_iterations = (max_stake_int - min_stake_int) / step_int + 1;

                            (0..stake_iterations)
                                .map(|idx| (min_stake_int + idx * step_int) as f64 / PRECISION as f64)
                                .collect()
                        };

                        // Генерируем комбинации для всех значений попыток
                        for stake_percent in stake_percent_range {
                            for attempts in params.min_attempts_count..=params.max_attempts_count {
                                combinations.push(ParameterCombination {
                                    num_low,
                                    search_threshold,
                                    high_threshold,
                                    payout_threshold,
                                    multiplier,
                                    stake_percent,
                                    attempts,
                                });

                                println!("Сгенерирована комбинация: множитель={:.1}, поиск={:.1}, ожидание={:.1}, ставка={:.1}",
                                         multiplier, search_threshold, high_threshold, payout_threshold);
                            }
                        }
                    }
                }
            }
        }
    }

    println!("Сгенерировано комбинаций: {}", combinations.len());
    combinations
}





fn strategy_triple_growth(
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
    attempts: u32,
    log_enabled: bool,
) -> (f64, f64, u32, u32, u32, u32) {
    let mut balance = initial_balance;
    let mut max_balance = initial_balance;

    // Добавляем более строгий режим логирования для отладки
    if log_enabled {
        println!("СТАРТ СТРАТЕГИИ: mult={:.1}, balance={:.2}, stake={:.2}, percent={:.2}%",
                 multiplier, initial_balance, stake, stake_percent);
    }

    // Предварительная проверка возможности выполнения серии ставок
    let mut test_balance = initial_balance;
    let mut test_stake = if bet_type == 0 {
        stake
    } else {
        initial_balance * (stake_percent / 100.0)
    };

    for i in 0..attempts {
        const EPSILON: f64 = 1e-6;
        if test_stake > test_balance * (1.0 + EPSILON) {
            if log_enabled {
                println!("ПРОВЕРКА НЕ ПРОЙДЕНА на попытке {}: stake={:.6} > balance={:.6} (diff={})",
                         i+1, test_stake, test_balance, test_stake - test_balance);
            }
            return (initial_balance, initial_balance, 0, 0, 0, 0);
        }
        test_balance -= test_stake;
        test_stake *= multiplier;
    }

    if log_enabled {
        println!("ПРОВЕРКА ПРОЙДЕНА: баланс после всех проверочных ставок: {:.6}", test_balance);
    }

    let mut total_series = 0u32;
    let mut winning_series = 0u32;
    let mut total_bets = 0u32;
    let mut consecutive_losses = 0u32;
    let len = numbers.len();
    let mut i = num_low;

    while i < len {
        if log_enabled && i % 500 == 0 {
            println!("Strategy: индекс {}, баланс {:.2}", i, balance);
        }

        // Добавьте здесь проверку на минимальный баланс:
        if balance < stake && bet_type == 0 {  // Для фиксированной ставки
            // Баланс меньше минимальной ставки, дальнейшие расчеты бессмысленны
            break;
        }
        if balance < initial_balance * (stake_percent / 100.0) && bet_type == 1 {  // Для процента
            // Баланс слишком мал для процентной ставки
            break;
        }

        // Остальная часть функции...
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
                let initial_bet = if bet_type == 0 {
                    stake
                } else {
                    balance * (stake_percent / 100.0)
                };

                if initial_bet > balance {
                    if log_enabled {
                        println!("Недостаточно средств на индексе {}. Баланс: {:.2}, требуется: {:.2}",
                                 i, balance, initial_bet);
                    }
                    break;
                }

                total_series += 1;
                let mut betting_attempts = 0;
                let mut current_i = search_i;
                let mut current_stake = initial_bet;

                while betting_attempts <= attempts - 1 && current_i < len - 1 {
                    if current_stake > balance {
                        break;
                    }

                    current_i += 1;
                    total_bets += 1;
                    balance -= current_stake;
                    if numbers[current_i] >= payout_threshold {
                        let win = current_stake * payout_threshold;
                        balance += win;
                        winning_series += 1;
                        consecutive_losses = 0;
                        max_balance = max_balance.max(balance);

                        if log_enabled && win > 100.0 {
                            println!("Strategy: крупный выигрыш на индексе {}, сумма {:.2}", current_i, win);
                        }
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

    (
        balance,
        max_balance,
        total_bets,
        total_series,
        winning_series,
        consecutive_losses,
    )
}pub fn optimize_parameters<F>(
    numbers: &Array1<f64>,
    params: &Params,
    progress_callback: F,
    cancel_flag: &Arc<AtomicBool>
) -> (Vec<OptimizationResult>, Option<std::time::Duration>, Option<std::time::Duration>)
where
    F: Fn(usize, usize) + Send + Sync + Clone + 'static,
{
    let cuda_available = if params.use_gpu {
        check_cuda_availability()
    } else {
        println!("GPU отключен в настройках.");
        false
    };

    let combinations = generate_parameter_combinations(params);
    let total_combinations = combinations.len();

    progress_callback(0, total_combinations);
    println!("Всего комбинаций для обработки: {}", total_combinations);

    // Создаем общую очередь задач для CPU и GPU
    let shared_task_queue = Arc::new(ArrayQueue::new(total_combinations));

    // Размер пакета данных для обработки - адаптируем под общее количество задач
    let batch_size = (BATCH_SIZE).min(total_combinations / 20);

    // Заполняем очередь пакетами задач
    let mut combination_index = 0;
    while combination_index < combinations.len() {
        let end_index = std::cmp::min(combination_index + batch_size, combinations.len());
        let batch = combinations[combination_index..end_index].to_vec();
        if shared_task_queue.push(batch).is_err() {
            // Если очередь заполнена, выходим из цикла
            break;
        }
        combination_index = end_index;
    }

    // Счетчик обработанных комбинаций (общий для CPU и GPU)
    let processed_count = Arc::new(AtomicUsize::new(0));
    // Убираем неиспользуемую переменную, добавляя нижнее подчеркивание
    let _progress_step = total_combinations / 100; // для обновления каждые 1%

    // Запускаем поток для отслеживания прогресса
    let progress_thread = {
        let processed_count_clone = Arc::clone(&processed_count);
        let cancel_flag_clone = Arc::clone(cancel_flag);
        let progress_callback_clone = progress_callback.clone();

        std::thread::spawn(move || {
            let mut last_update = std::time::Instant::now();
            let update_interval = std::time::Duration::from_millis(100);

            while !cancel_flag_clone.load(Ordering::SeqCst) {
                let now = std::time::Instant::now();
                if now.duration_since(last_update) >= update_interval {
                    let total_done = processed_count_clone.load(Ordering::SeqCst);

                    // Обновляем прогресс
                    progress_callback_clone(total_done, total_combinations);
                    last_update = now;
                }

                std::thread::sleep(std::time::Duration::from_millis(20));

                let total_done = processed_count_clone.load(Ordering::SeqCst);
                if total_done >= total_combinations {
                    break;
                }
            }
        })
    };

    let mut gpu_time = std::time::Duration::new(0, 0);
    let mut cpu_time = std::time::Duration::new(0, 0);

    let mut combined_results = Vec::new();

    // Запускаем GPU-обработчик, если доступен
    let gpu_thread = if cuda_available {
        let task_queue_gpu = Arc::clone(&shared_task_queue);
        let processed_count_clone = Arc::clone(&processed_count);
        let cancel_flag_clone = Arc::clone(cancel_flag);
        let numbers_clone = numbers.clone();
        let params_clone = params.clone();

        Some(std::thread::spawn(move || {
            let (results, time) = process_gpu_batches(
                &task_queue_gpu,
                &numbers_clone,
                &params_clone,
                &processed_count_clone,
                BLOCK_SIZE,
                GRID_SIZE,
                &cancel_flag_clone
            );
            (results, time)
        }))
    } else {
        None
    };

    // Запускаем CPU-обработчик
    let cpu_thread = {
        let task_queue_cpu = Arc::clone(&shared_task_queue);
        let processed_count_clone = Arc::clone(&processed_count);
        let cancel_flag_clone = Arc::clone(cancel_flag);
        let numbers_clone = numbers.clone();
        let params_clone = params.clone();

        std::thread::spawn(move || {
            let (results, time) = process_cpu_batches(
                &task_queue_cpu,
                &numbers_clone,
                &params_clone,
                &processed_count_clone,
                &cancel_flag_clone
            );
            (results, time)
        })
    };

    // Переменные для отслеживания количества результатов
    let mut cpu_results_len = 0;
    let mut gpu_results_len = 0;

    // Дожидаемся завершения потоков и собираем результаты
    if let Some(gpu_thread) = gpu_thread {
        if let Ok((results, time)) = gpu_thread.join() {
            gpu_results_len = results.len();
            combined_results.extend(results);  // Сразу добавляем результаты в общий список
            gpu_time = time;
        }
    }

    if let Ok((results, time)) = cpu_thread.join() {
        cpu_results_len = results.len();
        combined_results.extend(results);  // Сразу добавляем результаты в общий список
        cpu_time = time;

        // Сортируем результаты по прибыли (или другим критериям)
        combined_results.sort_by(|a, b| {
            b.profit.partial_cmp(&a.profit)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.balance.partial_cmp(&a.balance).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.num_low.cmp(&b.num_low))
        });

        // Ограничиваем количество результатов
        let max_results: usize = params.max_results.parse().unwrap_or(10000);
        combined_results.truncate(max_results);

        println!("\nСтатистика выполнения:");
        println!("CPU время: {:?}", cpu_time);
        println!("GPU время: {:?}", gpu_time);
        println!(
            "Найдено стратегий: {} (CPU: {}, GPU: {})",
            combined_results.len(),
            cpu_results_len,
            gpu_results_len
        );

        // Дожидаемся завершения потока прогресса
        if let Ok(_) = progress_thread.join() {
            progress_callback(total_combinations, total_combinations);
        }

        return (combined_results, Some(cpu_time), Some(gpu_time));
    }

    (Vec::new(), None, None)
}
fn process_cpu_batches(
    task_queue: &Arc<ArrayQueue<Vec<ParameterCombination>>>,
    numbers: &Array1<f64>,
    params: &Params,
    processed_count: &Arc<AtomicUsize>,
    cancel_flag: &Arc<AtomicBool>
) -> (Vec<OptimizationResult>, std::time::Duration) {
    let cpu_start = Instant::now();
    let mut cpu_results = Vec::new();

    // Собираем все задачи из очереди в один список для последовательной обработки
    let mut all_combinations = Vec::new();
    while let Some(batch) = task_queue.pop() {
        if cancel_flag.load(Ordering::SeqCst) {
            return (cpu_results, cpu_start.elapsed());
        }
        all_combinations.extend(batch);
    }

    println!("Собрано {} комбинаций для последовательной обработки", all_combinations.len());

    // Обрабатываем все комбинации последовательно
    for (idx, combo) in all_combinations.iter().enumerate() {
        if cancel_flag.load(Ordering::SeqCst) {
            break;
        }

        println!("Обработка комбинации #{} из {}: множитель={:.1}",
                 idx+1, all_combinations.len(), combo.multiplier);

        // Проверяем комбинацию напрямую, без использования пула потоков
        if let Some(result) = process_single_combination(combo, numbers, params) {
            cpu_results.push(result);
            println!("✓ Комбинация #{} ПРОШЛА проверку", idx+1);
        } else {
            println!("✗ Комбинация #{} НЕ ПРОШЛА проверку", idx+1);
        }

        // Обновляем счетчик прогресса после каждой комбинации
        processed_count.fetch_add(1, Ordering::SeqCst);
    }

    println!("Последовательная обработка завершена. Найдено {} прибыльных стратегий.", cpu_results.len());
    (cpu_results, cpu_start.elapsed())
}
fn process_gpu_batches(
    task_queue: &Arc<ArrayQueue<Vec<ParameterCombination>>>,
    numbers: &Array1<f64>,
    params: &Params,
    processed_count: &Arc<AtomicUsize>,
    block_size: u32,
    grid_size: u32,
    cancel_flag: &Arc<AtomicBool>
) -> (Vec<OptimizationResult>, std::time::Duration) {
    let gpu_start = Instant::now();
    let mut gpu_results = Vec::new();

    unsafe {
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();

        let _context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        ).unwrap();

        let ptx = CString::new(include_str!("../cuda/kernel.ptx")).unwrap();
        let module = Module::load_from_string(&ptx).unwrap();

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let numbers_slice = numbers.as_slice().unwrap();
        let kernel_name = CString::new("optimize_kernel").unwrap();
        let function = module.get_function(&kernel_name).unwrap();
        let bet_type = if params.bet_type == "fixed" { 0 } else { 1 };

        // Рассчитываем размер shared memory
        let shared_mem_size = std::cmp::min(
            numbers.len() * std::mem::size_of::<f64>(),
            4096 * std::mem::size_of::<f64>()
        );

        // Обрабатываем задачи из очереди
        while let Some(batch) = task_queue.pop() {
            if cancel_flag.load(Ordering::SeqCst) {
                return (gpu_results, gpu_start.elapsed());
            }

            // Подготавливаем данные для GPU
            let params_array: Vec<f64> = batch
                .iter()
                .flat_map(|combo| {
                    vec![
                        combo.num_low as f64,
                        combo.search_threshold,
                        combo.high_threshold,
                        combo.payout_threshold,
                        combo.multiplier,
                        combo.stake_percent,
                        combo.attempts as f64,
                    ]
                })
                .collect();

            let gpu_results_buffer = vec![
                GpuOptimizationResult {
                    balance: 0.0,
                    max_balance: 0.0,
                    total_bets: 0,
                    total_series: 0,
                    winning_series: 0,
                    profit: 0.0,
                    initial_balance: params.initial_balance,
                };
                batch.len()
            ];

            // Копируем данные на GPU
            let mut d_numbers = DeviceBuffer::from_slice(numbers_slice).unwrap();
            let mut d_params = DeviceBuffer::from_slice(&params_array).unwrap();
            let mut d_results = DeviceBuffer::from_slice(&gpu_results_buffer).unwrap();

            // Вычисляем параметры запуска ядра
            let num_blocks = (batch.len() as u32 + block_size - 1) / block_size;
            let actual_grid_size = std::cmp::min(num_blocks, grid_size);

            // Запускаем ядро
            launch!(function<<<(actual_grid_size, 1, 1), (block_size, 1, 1), shared_mem_size as u32, stream>>>(
                d_numbers.as_device_ptr(),
                d_params.as_device_ptr(),
                d_results.as_device_ptr(),
                numbers_slice.len() as i32,
                batch.len() as i32,
                bet_type,
                params.stake
            ))
                .unwrap();

            stream.synchronize().unwrap();

            // Копируем результаты обратно с GPU
            let mut batch_results = gpu_results_buffer;
            d_results.copy_to(&mut batch_results).unwrap();

            // Преобразуем результаты GPU в формат OptimizationResult
            let batch_results = batch_results
                .into_iter()
                .zip(batch.iter())
                .filter_map(|(gpu_result, combo)| {
                    if gpu_result.profit > 0.0 {
                        Some(OptimizationResult {
                            num_low: combo.num_low,
                            search_threshold: combo.search_threshold,
                            high_threshold: combo.high_threshold,
                            payout_threshold: combo.payout_threshold,
                            multiplier: combo.multiplier,
                            stake_percent: combo.stake_percent,
                            attempts: combo.attempts,
                            balance: gpu_result.balance,
                            max_balance: gpu_result.max_balance,
                            total_bets: gpu_result.total_bets as u32,
                            total_series: gpu_result.total_series as u32,
                            winning_series: gpu_result.winning_series as u32,
                            profit: gpu_result.profit,
                            initial_balance: gpu_result.initial_balance,
                            bet_type: params.bet_type.clone(),
                            initial_stake: params.stake,
                        })
                    } else {
                        None
                    }
                });

            // Добавляем результаты и обновляем счетчик прогресса
            gpu_results.extend(batch_results);
            processed_count.fetch_add(batch.len(), Ordering::SeqCst);
        }
    }

    (gpu_results, gpu_start.elapsed())
}
fn process_cpu_combinations(
    combinations: &[ParameterCombination],
    numbers: &Array1<f64>,
    params: &Params,
    cancel_flag: &Arc<AtomicBool>
) -> Vec<OptimizationResult> {
    // Проверяем флаг отмены в начале функции
    if cancel_flag.load(Ordering::SeqCst) {
        return Vec::new();
    }

    println!("Начинаем последовательную обработку {} комбинаций...", combinations.len());

    // Последовательно обрабатываем каждую комбинацию для диагностики проблемы
    let mut results = Vec::new();
    for (idx, combo) in combinations.iter().enumerate() {
        if cancel_flag.load(Ordering::SeqCst) {
            break;
        }

        println!("Обработка комбинации #{}: множитель={:.1}", idx+1, combo.multiplier);

        if let Some(result) = process_combination(combo, numbers, params) {
            results.push(result);
            println!("  -> ПРОШЛА ПРОВЕРКУ!");
        } else {
            println!("  -> НЕ ПРОШЛА ПРОВЕРКУ!");
        }
    }

    println!("Обработка завершена. Найдено {} прибыльных стратегий.", results.len());
    results
}

fn process_combination(
    combo: &ParameterCombination,
    numbers: &Array1<f64>,
    params: &Params,
) -> Option<OptimizationResult> {
    println!("Детальная проверка комбинации: множитель={:.3}, поиск={:.2}, ставка={:.2}",
             combo.multiplier, combo.search_threshold, combo.payout_threshold);

    let stake_value = if params.bet_type == "fixed" {
        params.stake
    } else {
        0.0
    };

    // Проверка возможности выполнить максимальное количество ставок с увеличенным допуском
    let initial_bet = if params.bet_type == "fixed" {
        params.stake
    } else {
        params.initial_balance * (combo.stake_percent / 100.0)
    };

    println!("  Начальная ставка: {:.6}, баланс: {:.2}", initial_bet, params.initial_balance);

    // Увеличим допуск при сравнении и добавим более подробное логирование
    const EPSILON: f64 = 1e-6;
    let mut test_balance = params.initial_balance;
    let mut test_stake = initial_bet;

    for i in 0..combo.attempts {
        if test_stake > test_balance * (1.0 + EPSILON) {
            println!("  Предварительная проверка НЕ ПРОЙДЕНА на попытке {}: stake={:.6} > balance={:.6} (diff={:.8})",
                     i+1, test_stake, test_balance, test_stake - test_balance);
            return None;
        }
        test_balance -= test_stake;
        test_stake *= combo.multiplier;
        println!("  Попытка {}: ставка стала {:.6}, баланс стал {:.6}", i+1, test_stake, test_balance);
    }

    println!("  Предварительная проверка ПРОЙДЕНА. Запуск полного расчета...");

    // Если проверка прошла, продолжаем с основным расчетом с включенным логированием
    let (balance, max_balance, total_bets, total_series, winning_series, _) =
        strategy_triple_growth(
            numbers,
            stake_value,
            combo.multiplier,
            params.initial_balance,
            combo.search_threshold,
            combo.high_threshold,
            combo.payout_threshold,
            combo.num_low,
            if params.bet_type == "fixed" { 0 } else { 1 },
            combo.stake_percent,
            combo.attempts as u32,
            true, // Включаем логирование
        );

    // Логирование результатов для отладки
    println!("  Результат: баланс={:.2}, прибыль={:.2}, прибыльная={}",
             balance, balance - params.initial_balance, balance > params.initial_balance);

    // Фильтруем только прибыльные стратегии с приемлемым допуском
    if total_series > 0 && balance > params.initial_balance * (1.0 + EPSILON) {
        Some(OptimizationResult {
            num_low: combo.num_low,
            search_threshold: combo.search_threshold,
            high_threshold: combo.high_threshold,
            payout_threshold: combo.payout_threshold,
            multiplier: combo.multiplier,
            stake_percent: combo.stake_percent,
            attempts: combo.attempts,
            balance,
            max_balance,
            total_bets,
            total_series,
            winning_series,
            profit: balance - params.initial_balance,
            initial_balance: params.initial_balance,
            bet_type: params.bet_type.clone(),
            initial_stake: params.stake,
        })
    } else {
        None
    }
}
fn process_single_combination(
    combo: &ParameterCombination,
    numbers: &Array1<f64>,
    params: &Params
) -> Option<OptimizationResult> {
    // Создаем копию чисел для изоляции вычислений
    let numbers_copy = numbers.clone();

    // Выполняем базовые проверки перед запуском основного алгоритма
    let stake_value = if params.bet_type == "fixed" {
        params.stake
    } else {
        0.0
    };

    let initial_bet = if params.bet_type == "fixed" {
        params.stake
    } else {
        params.initial_balance * (combo.stake_percent / 100.0)
    };

    // Предварительная проверка возможности выполнить серию ставок
    const EPSILON: f64 = 1e-6;
    let mut test_balance = params.initial_balance;
    let mut test_stake = initial_bet;

    for i in 0..combo.attempts {
        if test_stake > test_balance * (1.0 + EPSILON) {
            println!("  Множитель {:.1}: Предварительная проверка НЕ пройдена на попытке {} (ставка={:.6} > баланс={:.6})",
                     combo.multiplier, i+1, test_stake, test_balance);
            return None;
        }
        test_balance -= test_stake;
        test_stake *= combo.multiplier;
    }

    // Если предварительная проверка пройдена, выполняем основной расчет
    let (balance, max_balance, total_bets, total_series, winning_series, _) =
        strategy_triple_growth_isolated(
            &numbers_copy,
            stake_value,
            combo.multiplier,
            params.initial_balance,
            combo.search_threshold,
            combo.high_threshold,
            combo.payout_threshold,
            combo.num_low,
            if params.bet_type == "fixed" { 0 } else { 1 },
            combo.stake_percent,
            combo.attempts as u32
        );

    println!("  Множитель {:.1}: Результат расчета: баланс={:.2}, прибыль={:.2}, серий={}, выигрыш={}",
             combo.multiplier, balance, balance - params.initial_balance, total_series, winning_series);

    // Проверяем, является ли стратегия прибыльной
    if total_series > 0 && balance > params.initial_balance * (1.0 + EPSILON) {
        Some(OptimizationResult {
            num_low: combo.num_low,
            search_threshold: combo.search_threshold,
            high_threshold: combo.high_threshold,
            payout_threshold: combo.payout_threshold,
            multiplier: combo.multiplier,
            stake_percent: combo.stake_percent,
            attempts: combo.attempts,
            balance,
            max_balance,
            total_bets,
            total_series,
            winning_series,
            profit: balance - params.initial_balance,
            initial_balance: params.initial_balance,
            bet_type: params.bet_type.clone(),
            initial_stake: params.stake,
        })
    } else {
        None
    }
}

// Изолированная версия strategy_triple_growth, которая не зависит от внешнего состояния
fn strategy_triple_growth_isolated(
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

    // Тщательно проходим по всем числам, соблюдая оригинальную логику
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

    (
        balance,
        max_balance,
        total_bets,
        total_series,
        winning_series,
        consecutive_losses,
    )
}