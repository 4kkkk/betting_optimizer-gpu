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
//const P_THREADS: usize = 24;
// RTX 3060 поддерживает до 1024 потоков в блоке
const BLOCK_SIZE: u32 = 1024; // Увеличиваем до 512
const GRID_SIZE: u32 = 64;   // 64 блока

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
    let size_estimate = (params.max_num_low - params.min_num_low + 1)
        * ((params.max_search_threshold - params.min_search_threshold) / 0.1 + 1.0) as usize
        * ((params.max_high_threshold - params.min_high_threshold) / 0.1 + 1.0) as usize
        * ((params.max_payout_threshold - params.min_payout_threshold) / 0.1 + 1.0) as usize
        * ((params.max_multiplier - params.min_multiplier) / 0.1 + 1.0) as usize
        * (params.max_attempts_count - params.min_attempts_count + 1);

    let mut combinations = Vec::with_capacity(size_estimate);

    // Используем шаг 0.2 вместо 0.1 для уменьшения количества комбинаций
    let step = 0.1;

    for num_low in params.min_num_low..=params.max_num_low {
        let mut search_threshold = params.min_search_threshold;
        while search_threshold <= params.max_search_threshold {
            let mut high_threshold = params.min_high_threshold;
            while high_threshold <= params.max_high_threshold {
                let mut payout_threshold = params.min_payout_threshold;
                while payout_threshold <= params.max_payout_threshold {
                    let mut multiplier = params.min_multiplier;
                    while multiplier <= params.max_multiplier {
                        let stake_percent_range = if params.bet_type == "fixed" {
                            vec![params.min_stake_percent]
                        } else {
                            let mut percents = Vec::new();
                            let mut current = params.min_stake_percent;
                            while current <= params.max_stake_percent {
                                percents.push(current);
                                current += 0.1; // Увеличиваем шаг для уменьшения перебора
                            }
                            percents
                        };

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
                            }
                        }
                        multiplier += step;
                    }
                    payout_threshold += step;
                }
                high_threshold += step;
            }
            search_threshold += step;
        }
    }
    combinations
}

pub fn strategy_triple_growth(
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
) -> (f64, f64, u32, u32, u32, u32) {
    let mut test_balance = initial_balance;
    let mut test_stake = if bet_type == 0 {
        stake
    } else {
        initial_balance * (stake_percent / 100.0)
    };

    for _ in 0..attempts {
        if test_stake > test_balance {
            return (initial_balance, initial_balance, 0, 0, 0, 0);
        }
        test_balance -= test_stake;
        test_stake *= multiplier;
    }
    let mut balance = initial_balance;
    let mut max_balance = initial_balance;
    let mut total_series = 0u32;
    let mut winning_series = 0u32;
    let mut total_bets = 0u32;
    let mut consecutive_losses = 0u32;
    let len = numbers.len();
    let mut i = num_low;


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

pub fn optimize_parameters<F>(
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

    let gpu_percent = if cuda_available { 80 } else { 0 };
    let gpu_combinations = (total_combinations * gpu_percent) / 100;
    let _cpu_combinations = total_combinations - gpu_combinations;

    let progress_step = total_combinations / 20;
    let processed_cpu = Arc::new(AtomicUsize::new(0));
    let processed_gpu = Arc::new(AtomicUsize::new(0));

    let update_interval = std::time::Duration::from_millis(100);
    let progress_thread = {
        let processed_cpu_clone = Arc::clone(&processed_cpu);
        let processed_gpu_clone = Arc::clone(&processed_gpu);
        let cancel_flag_clone = Arc::clone(cancel_flag);
        let progress_callback_clone = progress_callback.clone();

        std::thread::spawn(move || {
            let mut last_update = std::time::Instant::now();

            while !cancel_flag_clone.load(Ordering::SeqCst) {
                let now = std::time::Instant::now();
                if now.duration_since(last_update) >= update_interval {
                    let cpu_done = processed_cpu_clone.load(Ordering::SeqCst);
                    let gpu_done = processed_gpu_clone.load(Ordering::SeqCst);
                    let total_done = cpu_done + gpu_done;

                    progress_callback_clone(total_done, total_combinations);

                    last_update = now;
                }

                std::thread::sleep(std::time::Duration::from_millis(20));

                let total_done = processed_cpu_clone.load(Ordering::SeqCst) + processed_gpu_clone.load(Ordering::SeqCst);
                if total_done >= total_combinations {
                    break;
                }
            }
        })
    };

    if total_combinations < 100 {
        let task_queue = Arc::new(ArrayQueue::new(combinations.len()));
        task_queue.push(combinations).unwrap();

        let gpu_results = if cuda_available {
            process_gpu_combinations(
                &task_queue,
                numbers,
                params,
                &processed_gpu,
                progress_step,
                total_combinations,
                BLOCK_SIZE,
                GRID_SIZE,
                cancel_flag
            )
        } else {
            (Vec::new(), std::time::Duration::new(0, 0))
        };

        let (mut results, gpu_time) = gpu_results;
        results.sort_by(|a, b| b.balance.partial_cmp(&a.balance).unwrap());

        let max_results: usize = params.max_results.parse().unwrap_or(10000);
        results.truncate(max_results);

        println!("\nСтатистика выполнения (только GPU):");
        println!("Время GPU: {:?}", gpu_time);
        println!("Найдено стратегий: {}", results.len());

        let cpu_start = Instant::now();

        if let Ok(_) = progress_thread.join() {
            progress_callback(total_combinations, total_combinations);
        }

        return (results, Some(cpu_start.elapsed()), Some(gpu_time));
    }

    let shared_task_queue = Arc::new(ArrayQueue::new(total_combinations));
    println!("Начинаем оптимизацию...");

    let batch_size = (BATCH_SIZE).min(total_combinations / 10);
    let gpu_batch_size = batch_size;
    let cpu_batch_size = batch_size / 2;

    for gpu_chunk in combinations[0..gpu_combinations].chunks(gpu_batch_size) {
        shared_task_queue.push(gpu_chunk.to_vec()).unwrap();
    }

    let cpu_combinations_slice = combinations[gpu_combinations..].to_vec();
    let task_queue_cpu = Arc::clone(&shared_task_queue);
    let numbers_for_cpu = numbers.clone();
    let params_for_cpu = params.clone();
    let processed_cpu_clone = Arc::clone(&processed_cpu);

    let cancel_flag_for_cpu = Arc::clone(cancel_flag);
    let cpu_thread = std::thread::spawn(move || {
        let cpu_start = Instant::now();
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        let mut cpu_results = Vec::new();
        let cpu_threads = params_for_cpu.cpu_threads;

        let p_core_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_threads)
            .start_handler(move |id| {
                if !core_ids.is_empty() && id < core_ids.len() {
                    core_affinity::set_for_current(core_ids[id]);
                }
                unsafe {
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
                }
            })
            .build()
            .unwrap();

        while let Some(chunk) = task_queue_cpu.pop() {
            if cancel_flag_for_cpu.load(Ordering::SeqCst) {
                return (cpu_results, cpu_start.elapsed());
            }

            let chunk_results = p_core_pool.install(|| {
                process_cpu_combinations(&chunk, &numbers_for_cpu, &params_for_cpu, &cancel_flag_for_cpu)
            });

            cpu_results.extend(chunk_results);

            let current = processed_cpu_clone.fetch_add(chunk.len(), Ordering::SeqCst);
            if current / progress_step != (current + chunk.len()) / progress_step {
                println!("CPU прогресс: {}%",
                         ((current + chunk.len()) * 100) / total_combinations);
            }
        }

        for cpu_chunk in cpu_combinations_slice.chunks(cpu_batch_size) {
            let chunk_results = p_core_pool.install(|| {
                process_cpu_combinations(cpu_chunk, &numbers_for_cpu, &params_for_cpu, &cancel_flag_for_cpu)

            });
            cpu_results.extend(chunk_results);

            let current = processed_cpu_clone.fetch_add(cpu_chunk.len(), Ordering::SeqCst);
            if current / progress_step != (current + cpu_chunk.len()) / progress_step {
                println!("CPU прогресс: {}%",
                         ((current + cpu_chunk.len()) * 100) / total_combinations);
            }
        }

        (cpu_results, cpu_start.elapsed())
    });

    let gpu_results = if cuda_available {
        process_gpu_combinations(
            &shared_task_queue,
            numbers,
            params,
            &processed_gpu,
            progress_step,
            total_combinations,
            BLOCK_SIZE,
            GRID_SIZE,
            cancel_flag
        )
    } else {
        (Vec::new(), std::time::Duration::new(0, 0))
    };

    let (cpu_results, cpu_time) = cpu_thread.join().unwrap_or_else(|_| {
        println!("Ошибка при получении результатов CPU");
        (Vec::new(), std::time::Duration::new(0, 0))
    });
    let (gpu_results, gpu_time) = gpu_results;

    let cpu_results_len = cpu_results.len();
    let gpu_results_len = gpu_results.len();

    let mut combined_results = Vec::with_capacity(cpu_results_len + gpu_results_len);
    combined_results.extend(cpu_results);
    combined_results.extend(gpu_results);

    combined_results.sort_by(|a, b| {
        b.profit.partial_cmp(&a.profit)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.balance.partial_cmp(&a.balance).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.num_low.cmp(&b.num_low)) // Добавьте дополнительные критерии
            .then_with(|| a.search_threshold.partial_cmp(&b.search_threshold).unwrap_or(std::cmp::Ordering::Equal))
    });

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

    if let Ok(_) = progress_thread.join() {
        progress_callback(total_combinations, total_combinations);
    }

    (combined_results, Some(cpu_time), Some(gpu_time))
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

    // Оптимизация для многоядерных процессоров
    let cpu_threads = params.cpu_threads;
    let chunk_size = std::cmp::max(1, combinations.len() / cpu_threads);

    let pool_result = rayon::ThreadPoolBuilder::new()
        .num_threads(cpu_threads)
        .build();

    match pool_result {
        Ok(pool) => {
            pool.install(|| {
                combinations
                    .par_chunks(chunk_size)
                    .flat_map_iter(|chunk| {
                        // Проверка отмены
                        if cancel_flag.load(Ordering::SeqCst) {
                            return vec![]; // Возвращаем пустой вектор если отменено
                        }

                        chunk
                            .iter()
                            .filter_map(|combo| process_combination(combo, numbers, params))
                            .collect() // Добавить это
                    })
                    .collect()
            })
        },
        Err(_) => {
            // Если не удалось создать пул
            combinations
                .par_chunks(chunk_size)
                .flat_map_iter(|chunk| {
                    // Проверка отмены
                    if cancel_flag.load(Ordering::SeqCst) {
                        return vec![];
                    }

                    chunk
                        .iter()
                        .filter_map(|combo| process_combination(combo, numbers, params))
                        .collect() // Добавить это
                })
                .collect()
        }
    }
}



fn process_gpu_combinations(
    task_queue: &Arc<ArrayQueue<Vec<ParameterCombination>>>,
    numbers: &Array1<f64>,
    params: &Params,
    processed_gpu: &Arc<AtomicUsize>,
    progress_step: usize,
    total_combinations: usize,
    block_size: u32,
    grid_size: u32,
    cancel_flag: &Arc<AtomicBool>
) -> (Vec<OptimizationResult>, std::time::Duration) {
    if cancel_flag.load(Ordering::SeqCst) {
        return (Vec::new(), std::time::Duration::new(0, 0));
    }
    let gpu_start = Instant::now();
    let mut all_gpu_results = Vec::new();

    // Рассчитываем оптимальный размер shared memory
    let shared_mem_size = std::cmp::min(
        numbers.len() * size_of::<f64>(),
        4096 * size_of::<f64>() // Максимум 4096 элементов
    );

    unsafe {
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();

        // Удаляем неработающую строку с set_attribute
        // device.set_attribute(DeviceAttribute::GpuDirectRdmaFlushWrites, 1).unwrap_or_default();

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

        while let Some(batch) = task_queue.pop() {
            if cancel_flag.load(Ordering::SeqCst) {
                return (all_gpu_results, gpu_start.elapsed());
            }
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

            let mut d_numbers = DeviceBuffer::from_slice(numbers_slice).unwrap();
            let mut d_params = DeviceBuffer::from_slice(&params_array).unwrap();
            let mut d_results = DeviceBuffer::from_slice(&gpu_results_buffer).unwrap();

            // Расчет оптимальных параметров
            let num_blocks = (batch.len() as u32 + block_size - 1) / block_size;
            let actual_grid_size = std::cmp::min(num_blocks, grid_size);

            // Исправляем ошибку типа для shared_mem_size
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

            let mut batch_results = gpu_results_buffer;
            d_results.copy_to(&mut batch_results).unwrap();
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

            all_gpu_results.extend(batch_results);

            let current = processed_gpu.fetch_add(batch.len(), Ordering::SeqCst);
            if current / progress_step != (current + batch.len()) / progress_step {
                println!("GPU прогресс: {}%",
                         ((current + batch.len()) * 100) / total_combinations);
            }
            if current / progress_step != (current + batch.len()) / progress_step {
                println!("GPU прогресс: {}%",
                         ((current + batch.len()) * 100) / total_combinations);
            }
        }
    }

    (all_gpu_results, gpu_start.elapsed())
}

fn process_combination(
    combo: &ParameterCombination,
    numbers: &Array1<f64>,
    params: &Params,
) -> Option<OptimizationResult> {
    let stake_value = if params.bet_type == "fixed" {
        params.stake
    } else {
        0.0
    };

    // Проверка возможности выполнить максимальное количество ставок
    let initial_bet = if params.bet_type == "fixed" {
        params.stake
    } else {
        params.initial_balance * (combo.stake_percent / 100.0)
    };

    // Быстрая проверка перед выполнением полного расчета
    let mut test_balance = params.initial_balance;
    let mut test_stake = initial_bet;
    for _ in 0..combo.attempts {
        if test_stake > test_balance {
           return None;
        }
        test_balance -= test_stake;
        test_stake *= combo.multiplier;
    }

    // Если проверка прошла, продолжаем с основным расчетом
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
        );

    // Фильтруем только прибыльные стратегии
    if total_series > 0 && balance > params.initial_balance {
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