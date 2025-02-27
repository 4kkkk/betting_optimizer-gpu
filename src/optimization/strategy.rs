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
use std::mem::size_of;











fn get_optimal_threads() -> usize {
    let physical_cores = num_cpus::get_physical();
    let logical_cores = num_cpus::get();

    if physical_cores >= 16 {
        physical_cores - 1
    } else if logical_cores >= 24 {
        logical_cores - 4
    } else if physical_cores > 4 {
        physical_cores - 2
    } else {
        physical_cores
    }
}

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

                while betting_attempts < attempts && current_i < len - 1 {
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

pub fn optimize_parameters(numbers: &Array1<f64>, params: &Params) -> Vec<OptimizationResult> {
    let cuda_available = check_cuda_availability();
    let combinations = generate_parameter_combinations(params);
    let total_combinations = combinations.len();

    let cpu_gpu_ratio = if cuda_available {
        0.40
    } else {
        1.0
    };

    let progress_step = total_combinations / 20;
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

    println!("Всего комбинаций для обработки: {}", total_combinations);
    println!("Начинаем оптимизацию...");

    let cpu_combinations_count = if cuda_available {
        (total_combinations as f64 * cpu_gpu_ratio) as usize
    } else {
        total_combinations
    };
    let gpu_combinations_count = total_combinations - cpu_combinations_count;

    println!("Распределение работы: CPU: {}, GPU: {}",
             cpu_combinations_count, gpu_combinations_count);

    let cpu_queue = Arc::new(ArrayQueue::new(cpu_combinations_count + 1000));
    let gpu_queue = Arc::new(ArrayQueue::new(gpu_combinations_count + 1000));

    let mut sorted_combinations = combinations;
    sorted_combinations.sort_by(|a, b| {
        let a_weight = a.attempts as f64 * a.multiplier;
        let b_weight = b.attempts as f64 * b.multiplier;
        b_weight.partial_cmp(&a_weight).unwrap()
    });

    let mut gpu_count = 0;
    let mut _cpu_count = 0;

    for chunk in sorted_combinations.chunks(1000) {
        if gpu_count < gpu_combinations_count && cuda_available {
            let can_take = std::cmp::min(chunk.len(), gpu_combinations_count - gpu_count);

            if can_take > 0 {
                let (gpu_chunk, rest) = if can_take < chunk.len() {
                    let mut gpu_part = chunk.to_vec();
                    let cpu_part = gpu_part.split_off(can_take);
                    (gpu_part, Some(cpu_part))
                } else {
                    (chunk.to_vec(), None)
                };

                gpu_queue.push(gpu_chunk).unwrap();
                gpu_count += can_take;

                if let Some(cpu_chunk) = rest {
                    let chunk_len = cpu_chunk.len();
                    cpu_queue.push(cpu_chunk).unwrap();
                    _cpu_count += chunk_len;
                }
            }
        } else {
            cpu_queue.push(chunk.to_vec()).unwrap();
            _cpu_count += chunk.len();
        }
    }

    let numbers_for_cpu = numbers.clone();
    let params_for_cpu = params.clone();
    let processed_cpu_clone = Arc::clone(&processed_cpu);

    let cpu_thread = std::thread::spawn(move || {
        let cpu_start = Instant::now();
        let core_ids = core_affinity::get_core_ids().unwrap();
        let mut cpu_results = Vec::new();

        let p_threads = get_optimal_threads();
        println!("CPU использует {} потоков", p_threads);

        let p_core_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(p_threads)
            .start_handler(move |id| {
                if id < core_ids.len() {
                    core_affinity::set_for_current(core_ids[id]);
                }
                unsafe {
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
                }
            })
            .build()
            .unwrap();

        while let Some(chunk) = cpu_queue.pop() {
            let chunk_results = p_core_pool.install(|| {
                process_cpu_combinations(&chunk, &numbers_for_cpu, &params_for_cpu)
            });
            cpu_results.extend(chunk_results);

            let current = processed_cpu_clone.fetch_add(chunk.len(), Ordering::SeqCst);
            if current / progress_step != (current + chunk.len()) / progress_step {
                println!("CPU прогресс: {}%",
                         ((current + chunk.len()) as f64 * 100.0) / total_combinations as f64);
            }
        }
        (cpu_results, cpu_start.elapsed())
    });

    let gpu_results = if cuda_available {
        process_gpu_combinations(
            &gpu_queue,
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

    let mut combined_results = Vec::with_capacity(cpu_results.len() + gpu_results.len());
    combined_results.extend(cpu_results.clone());
    combined_results.extend(gpu_results.clone());
    combined_results.sort_by(|a, b| b.profit.partial_cmp(&a.profit).unwrap());

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
    let core_ids = core_affinity::get_core_ids().unwrap();

    let p_threads = get_optimal_threads();

    let chunk_size = std::cmp::max(1, combinations.len() / (p_threads * 2));

    let p_core_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(p_threads)
        .start_handler(move |id| {
            if id < core_ids.len() {
                core_affinity::set_for_current(core_ids[id % core_ids.len()]);
            }
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
                    .filter_map(|combo| {
                        if combo.multiplier > 3.0 && combo.attempts > 5 {
                            if combo.payout_threshold < 2.0 {
                                return None;
                            }
                        }

                        process_combination(combo, numbers, params)
                    })
            })
            .collect()
    })
}

fn process_gpu_combinations(
    task_queue: &Arc<ArrayQueue<Vec<ParameterCombination>>>,
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

        // Фиксированные значения для RTX 3060 (12GB)
        let total_mem = 12_usize * 1024 * 1024 * 1024; // 12GB
        let free_mem = (total_mem as f64 * 0.8) as usize; // 80% свободно

        println!("GPU память: приблизительно свободно {} MB / всего {} MB",
                 free_mem / (1024 * 1024),
                 total_mem / (1024 * 1024));

        let _context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        ).unwrap();

        let ptx = CString::new(include_str!("../cuda/kernel.ptx")).unwrap();
        let module = Module::load_from_string(&ptx).unwrap();
        let numbers_slice = numbers.as_slice().unwrap();
        let kernel_name = CString::new("optimize_kernel").unwrap();
        let function = module.get_function(&kernel_name).unwrap();
        let bet_type = if params.bet_type == "fixed" { 0 } else { 1 };

        let element_size = size_of::<f64>() * 7;
        let numbers_size = size_of::<f64>() * numbers_slice.len();
        let result_size = size_of::<GpuOptimizationResult>();

        let available_memory = (free_mem as f64 * 0.85) as usize;
        let max_batch_size = (available_memory - numbers_size) / (element_size + result_size);

        println!("Максимальный размер GPU пакета: {}", max_batch_size);

        // Определяем число потоков на основе вашей видеокарты
        let num_streams = 7; // Для RTX 3060 оптимально ~7 потоков

        let mut streams = Vec::with_capacity(num_streams);
        for _ in 0..num_streams {
            streams.push(Stream::new(StreamFlags::NON_BLOCKING, None).unwrap());
        }

        println!("GPU: используем {} CUDA потоков", num_streams);
        let mut stream_idx = 0;
        let mut batch_count = 0;

        while let Some(mut batch) = task_queue.pop() {
            batch_count += 1;

            if batch.len() > max_batch_size {
                let remaining = batch.split_off(max_batch_size);
                task_queue.push(remaining).unwrap();
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

            let threads_per_block = 256;

            let num_blocks = (batch.len() as u32 + threads_per_block - 1) / threads_per_block;

            let mut d_numbers = DeviceBuffer::from_slice(numbers_slice).unwrap();
            let mut d_params = DeviceBuffer::from_slice(&params_array).unwrap();
            let mut d_results = DeviceBuffer::from_slice(&gpu_results_buffer).unwrap();

            let current_stream = &streams[stream_idx];

            launch!(function<<<(num_blocks, 1, 1), (threads_per_block, 1, 1), 0, current_stream>>>(
                d_numbers.as_device_ptr(),
                d_params.as_device_ptr(),
                d_results.as_device_ptr(),
                numbers_slice.len() as i32,
                batch.len() as i32,
                bet_type,
                params.stake
            ))
                .unwrap();

            stream_idx = (stream_idx + 1) % num_streams;

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

            if batch_count % 10 == 0 {
                // Периодически освобождаем память GPU
                if batch_count % 30 == 0 {
                    println!("GPU: плановая очистка памяти");
                    rustacuda::context::ContextStack::pop().unwrap();
                    rustacuda::context::ContextStack::push(&_context).unwrap();
                }
            }

            all_gpu_results.extend(batch_results);

            let current = processed_gpu.fetch_add(batch.len(), Ordering::SeqCst);
            if current / progress_step != (current + batch.len()) / progress_step {
                println!("GPU прогресс: {}%",
                         ((current + batch.len()) as f64 * 100.0) / total_combinations as f64);

                if current > 0 {
                    let elapsed = gpu_start.elapsed().as_secs_f64();
                    let combinations_per_sec = current as f64 / elapsed;
                    let estimated_remaining = (total_combinations - current) as f64 / combinations_per_sec;

                    println!("    Скорость: {:.1} комбинаций/сек", combinations_per_sec);
                    println!("    Примерное оставшееся время: {:.1} мин", estimated_remaining / 60.0);
                }
            }
        }

        for stream in &streams {
            stream.synchronize().unwrap();
        }
    }

    (all_gpu_results, gpu_start.elapsed())
}


fn process_combination(
    combo: &ParameterCombination,
    numbers: &Array1<f64>,
    params: &Params,
) -> Option<OptimizationResult> {
    let max_possible_win_ratio = combo.payout_threshold / combo.multiplier;
    let effective_attempts = combo.attempts as f64 * max_possible_win_ratio;

    if effective_attempts < 1.5 {
        return None;
    }

    let stake_value = if params.bet_type == "fixed" {
        params.stake
    } else {
        0.0
    };

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

    let roi = if total_bets > 0 {
        (balance - params.initial_balance) / (total_bets as f64 * params.stake)
    } else {
        0.0
    };

    if total_series > 0 && balance > params.initial_balance && roi > 0.01 {
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
