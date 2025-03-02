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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use windows_sys::Win32::System::Threading::{
    GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST,
};

const P_THREADS: usize = 23;

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
fn round_to_cents(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}


fn generate_parameter_combinations(params: &Params) -> Vec<ParameterCombination> {
    let size_estimate = (params.max_num_low - params.min_num_low + 1)
        * ((params.max_search_threshold - params.min_search_threshold) / 0.1 + 1.0) as usize
        * ((params.max_high_threshold - params.min_high_threshold) / 0.1 + 1.0) as usize
        * ((params.max_payout_threshold - params.min_payout_threshold) / 0.1 + 1.0) as usize
        * ((params.max_multiplier - params.min_multiplier) / 0.1 + 1.0) as usize
        * (params.max_attempts_count - params.min_attempts_count + 1);

    let mut combinations = Vec::with_capacity(size_estimate);


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
                                current += 0.1;
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
                        multiplier += 0.1;
                    }
                    payout_threshold += 0.1;
                }
                high_threshold += 0.1;
            }
            search_threshold += 0.1;
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
    let mut balance = round_to_cents(initial_balance);
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
                    round_to_cents(stake)
                } else {
                    round_to_cents(balance * (stake_percent / 100.0))
                };

                let mut current_stake = initial_bet;

                while betting_attempts <= attempts - 1 && current_i < len - 1 {
                    // Проверка достаточности баланса для текущей ставки
                    if current_stake > balance {
                        // Если баланса не хватает, прерываем серию ставок
                        break;
                    }

                    current_i += 1;
                    total_bets += 1;
                    balance = round_to_cents(balance - current_stake);

                    if numbers[current_i] >= payout_threshold {
                        let win = round_to_cents(current_stake * payout_threshold);
                        balance = round_to_cents(balance + win);
                        winning_series += 1;
                        consecutive_losses = 0;
                        max_balance = round_to_cents(max_balance.max(balance));
                        break;
                    } else {
                        consecutive_losses += 1;
                        current_stake = round_to_cents(current_stake * multiplier);
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
        round_to_cents(balance),
        round_to_cents(max_balance),
        total_bets,
        total_series,
        winning_series,
        consecutive_losses,
    )
}
pub fn optimize_parameters(numbers: &Array1<f64>, params: &Params) -> Vec<OptimizationResult> {
    let cuda_available = check_cuda_availability();
    let combinations = generate_parameter_combinations(params);
    let total_combinations = combinations.len();
    let progress_step = std::cmp::max(1, total_combinations / 20); // 5% шаг, минимум 1
    let processed_cpu = Arc::new(AtomicUsize::new(0));
    let processed_gpu = Arc::new(AtomicUsize::new(0));

    if combinations.len() < 100 {
        let task_queue = Arc::new(ArrayQueue::new(combinations.len()));
        task_queue.push(combinations).unwrap();

        let gpu_results = if cuda_available {
            process_gpu_combinations(
                &task_queue,
                numbers,
                params,
                &processed_gpu,
                progress_step,
                total_combinations
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

        return results;
    }

    let shared_task_queue = Arc::new(ArrayQueue::new(combinations.len()));
    println!("Всего комбинаций для обработки: {}", total_combinations);
    println!("Начинаем оптимизацию...");

    // Оптимизированный размер пакета для RTX 3060 12GB
    let optimal_batch_size = 8000; // Увеличен размер пакета для лучшего использования 12GB VRAM
    println!("Используем оптимизированный размер пакета для GPU: {}", optimal_batch_size);

    for chunk in combinations.chunks(optimal_batch_size) {
        shared_task_queue.push(chunk.to_vec()).unwrap();
    }

    let task_queue_cpu = Arc::clone(&shared_task_queue);
    let numbers_for_cpu = numbers.clone();
    let params_for_cpu = params.clone();
    let processed_cpu_clone = Arc::clone(&processed_cpu);

    let cpu_thread = std::thread::spawn(move || {
        let cpu_start = Instant::now();
        let core_ids = core_affinity::get_core_ids().unwrap();
        let mut cpu_results = Vec::new();

        let p_core_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(P_THREADS)
            .start_handler(move |id| {
                core_affinity::set_for_current(core_ids[id]);
                unsafe {
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
                }
            })
            .build()
            .unwrap();

        while let Some(chunk) = task_queue_cpu.pop() {
            let chunk_results = p_core_pool.install(|| {
                process_cpu_combinations(&chunk, &numbers_for_cpu, &params_for_cpu)
            });
            cpu_results.extend(chunk_results);

            let current = processed_cpu_clone.fetch_add(chunk.len(), Ordering::SeqCst);
            if current / progress_step != (current + chunk.len()) / progress_step {
                println!("CPU прогресс: {}%",
                         ((current + chunk.len()) * 100) / total_combinations);
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
            total_combinations
        )
    } else {
        (Vec::new(), std::time::Duration::new(0, 0))
    };

    let (cpu_results, cpu_time) = cpu_thread.join().unwrap();
    let (gpu_results, gpu_time) = gpu_results;

    let mut combined_results = Vec::new();
    combined_results.extend(cpu_results.clone());
    combined_results.extend(gpu_results.clone());
    combined_results.sort_by(|a, b| b.balance.partial_cmp(&a.balance).unwrap());

    let max_results: usize = params.max_results.parse().unwrap_or(10000);
    combined_results.truncate(max_results);

    println!("\nСтатистика выполнения:");
    println!("CPU время: {:?}", cpu_time);
    println!("GPU время: {:?}", gpu_time);
    println!(
        "Найдено стратегий: {} (CPU: {}, GPU: {})",
        combined_results.len(),
        cpu_results.len(),
        gpu_results.len()
    );

    combined_results
}

fn process_cpu_combinations(
    combinations: &[ParameterCombination],
    numbers: &Array1<f64>,
    params: &Params,
) -> Vec<OptimizationResult> {
    // Используем просто core_affinity вместо hwloc
    let core_ids = core_affinity::get_core_ids().unwrap();

    // Вычисляем размер чанка и гарантируем что он не будет нулевым
    let chunk_size = std::cmp::max(1, combinations.len() / P_THREADS);

    let p_core_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(P_THREADS)
        .start_handler(move |id| {
            core_affinity::set_for_current(core_ids[id]);
            unsafe {
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            }
        })
        .build()
        .unwrap();

    p_core_pool.install(|| {
        combinations
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                chunk
                    .iter()
                    .filter_map(|combo| process_combination(combo, numbers, params))
            })
            .collect()
    })
}

fn process_gpu_combinations(    task_queue: &Arc<ArrayQueue<Vec<ParameterCombination>>>,
    numbers: &Array1<f64>,
    params: &Params,
    processed_gpu: &Arc<AtomicUsize>,
    progress_step: usize,
    total_combinations: usize,
) -> (Vec<OptimizationResult>, std::time::Duration) {
    let gpu_start = Instant::now();
    let mut all_gpu_results = Vec::new();

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

        while let Some(batch) = task_queue.pop() {
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
                    initial_balance: round_to_cents(params.initial_balance),
                };
                batch.len()
            ];

            let mut d_numbers = DeviceBuffer::from_slice(numbers_slice).unwrap();
            let mut d_params = DeviceBuffer::from_slice(&params_array).unwrap();
            let mut d_results = DeviceBuffer::from_slice(&gpu_results_buffer).unwrap();

            launch!(function<<<(batch.len() as u32, 1, 1), (1024, 1, 1), 0, stream>>>(
                d_numbers.as_device_ptr(),
                d_params.as_device_ptr(),
                d_results.as_device_ptr(),
                numbers_slice.len() as i32,
                batch.len() as i32,
                bet_type,
                round_to_cents(params.stake)
            ))
                .unwrap();

            stream.synchronize().unwrap();

            let mut batch_results = gpu_results_buffer;
            d_results.copy_to(&mut batch_results).unwrap();
            let batch_results = batch_results
                .into_iter()
                .zip(batch.iter())
                .filter_map(|(gpu_result, combo)| {
                    // Отфильтровываем комбинации с недостаточным балансом для всех ставок
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
                            initial_stake: round_to_cents(params.stake),
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
        round_to_cents(params.stake)
    } else {
        0.0
    };

    // Проверка возможности выполнить максимальное количество ставок
    let initial_bet = if params.bet_type == "fixed" {
        round_to_cents(params.stake)
    } else {
        round_to_cents(params.initial_balance * (combo.stake_percent / 100.0))
    };

    // Симуляция последовательности ставок для проверки достаточности баланса
    let mut test_balance = round_to_cents(params.initial_balance);
    let mut test_stake = initial_bet;
    for _ in 0..combo.attempts {
        if test_stake > test_balance {
            // Если для следующей ставки недостаточно средств, отбраковываем эту комбинацию
            return None;
        }
        test_balance = round_to_cents(test_balance - test_stake);
        test_stake = round_to_cents(test_stake * combo.multiplier);
    }

    // Если проверка прошла, продолжаем с основным расчетом
    let (balance, max_balance, total_bets, total_series, winning_series, _) =
        strategy_triple_growth(
            numbers,
            stake_value,
            combo.multiplier,
            round_to_cents(params.initial_balance),
            combo.search_threshold,
            combo.high_threshold,
            combo.payout_threshold,
            combo.num_low,
            if params.bet_type == "fixed" { 0 } else { 1 },
            combo.stake_percent,
            combo.attempts as u32,
        );

    if total_series > 0 && balance > round_to_cents(params.initial_balance) {
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
            profit: round_to_cents(balance - round_to_cents(params.initial_balance)),  // Округляем прибыль
            initial_balance: round_to_cents(params.initial_balance),  // Округляем начальный баланс
            bet_type: params.bet_type.clone(),
            initial_stake: round_to_cents(params.stake),  // Округляем начальную ставку
        })
    } else {
        None
    }
}