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
use std::io;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use std::fs::{File, create_dir_all};
use std::path::Path;
use eframe::emath::OrderedFloat;
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

struct ParameterCombinationIterator {
    params: Params,
    indices: [usize; 7], // Индексы для [num_low, search_threshold, high_threshold, payout_threshold, multiplier, stake_percent, attempts]
    current_values: [f64; 7], // Текущие значения параметров
    finished: bool,
    max_indices: [usize; 7], // Максимальные индексы для каждого параметра
    stake_percent_values: Vec<f64>, // Кэшируем значения stake_percent
}

impl ParameterCombinationIterator {
    fn new(params: Params) -> Self {

        let max_num_low = params.max_num_low - params.min_num_low;
        let max_search = ((params.max_search_threshold - params.min_search_threshold) / 0.1) as usize;
        let max_high = ((params.max_high_threshold - params.min_high_threshold) / 0.1) as usize;
        let max_payout = ((params.max_payout_threshold - params.min_payout_threshold) / 0.1) as usize;
        let max_multi = ((params.max_multiplier - params.min_multiplier) / 0.1) as usize;
        let max_attempts = params.max_attempts_count - params.min_attempts_count;

        let mut stake_percents = Vec::new();
        if params.bet_type == "fixed" {
            stake_percents.push(params.min_stake_percent);
        } else {
            let mut current = params.min_stake_percent;
            while current <= params.max_stake_percent {
                stake_percents.push(current);
                current += 0.1;
            }
        }
        let max_stake = stake_percents.len() - 1;

        let current_values = [
            params.min_num_low as f64,
            params.min_search_threshold,
            params.min_high_threshold,
            params.min_payout_threshold,
            params.min_multiplier,
            stake_percents[0],
            params.min_attempts_count as f64,
        ];

        ParameterCombinationIterator {
            params,
            indices: [0, 0, 0, 0, 0, 0, 0],
            current_values,
            finished: false,
            max_indices: [max_num_low, max_search, max_high, max_payout, max_multi, max_stake, max_attempts],
            stake_percent_values: stake_percents,
        }
    }

    fn estimate_total_count(&self) -> usize {

        let mut total = 1;
        for i in 0..7 {
            total *= self.max_indices[i] + 1;
        }
        total
    }

    fn create_combination(&self) -> ParameterCombination {
        ParameterCombination {
            num_low: self.current_values[0] as usize,
            search_threshold: self.current_values[1],
            high_threshold: self.current_values[2],
            payout_threshold: self.current_values[3],
            multiplier: self.current_values[4],
            stake_percent: self.current_values[5],
            attempts: self.current_values[6] as usize,
        }
    }
}

impl Iterator for ParameterCombinationIterator {
    type Item = ParameterCombination;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let combination = self.create_combination();

        let mut i = 6; // Начинаем с последнего индекса
        loop {
            if i == 5 { // Особая обработка для stake_percent
                if self.indices[i] < self.max_indices[i] {
                    self.indices[i] += 1;
                    self.current_values[i] = self.stake_percent_values[self.indices[i]];
                    break;
                } else {
                    self.indices[i] = 0;
                    self.current_values[i] = self.stake_percent_values[0];
                    i -= 1;
                }
            } else if i == 0 { // num_low
                if self.indices[i] < self.max_indices[i] {
                    self.indices[i] += 1;
                    self.current_values[i] = (self.params.min_num_low + self.indices[i]) as f64;
                    break;
                } else {

                    self.finished = true;
                    return Some(combination);
                }
            } else if i == 6 { // attempts
                if self.indices[i] < self.max_indices[i] {
                    self.indices[i] += 1;
                    self.current_values[i] = (self.params.min_attempts_count + self.indices[i]) as f64;
                    break;
                } else {
                    self.indices[i] = 0;
                    self.current_values[i] = self.params.min_attempts_count as f64;
                    i -= 1;
                }
            } else { // Остальные параметры с шагом 0.1
                if self.indices[i] < self.max_indices[i] {
                    self.indices[i] += 1;
                    match i {
                        1 => self.current_values[i] = self.params.min_search_threshold + (self.indices[i] as f64 * 0.1),
                        2 => self.current_values[i] = self.params.min_high_threshold + (self.indices[i] as f64 * 0.1),
                        3 => self.current_values[i] = self.params.min_payout_threshold + (self.indices[i] as f64 * 0.1),
                        4 => self.current_values[i] = self.params.min_multiplier + (self.indices[i] as f64 * 0.1),
                        _ => unreachable!(),
                    }
                    break;
                } else {
                    self.indices[i] = 0;
                    match i {
                        1 => self.current_values[i] = self.params.min_search_threshold,
                        2 => self.current_values[i] = self.params.min_high_threshold,
                        3 => self.current_values[i] = self.params.min_payout_threshold,
                        4 => self.current_values[i] = self.params.min_multiplier,
                        _ => unreachable!(),
                    }
                    i -= 1;
                }
            }
        }

        Some(combination)
    }
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
    
                    if current_stake > balance {
            
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
pub fn optimize_parameters(numbers: &Array1<f64>, params: &Params) -> Result<Vec<OptimizationResult>, io::Error> {
    let cuda_available = check_cuda_availability();


    let param_iterator = ParameterCombinationIterator::new(params.clone());
    let total_combinations = param_iterator.estimate_total_count();

    println!("Оценка количества комбинаций: {}", total_combinations);
    println!("Начинаем оптимизацию с итеративной обработкой...");

    let search_mode = match &params.search_mode {
        Some(mode) => mode.as_str(),
        None => "chunked",
    };
    
    let max_batch_size: usize = params.max_combination_batch
        .parse()
        .unwrap_or(1000000);

    let progress_step = std::cmp::max(1, total_combinations / 20);
    let processed_count = Arc::new(AtomicUsize::new(0));

    if search_mode == "full" && total_combinations < 100 {

        let mut all_combinations = Vec::new();
        for combo in param_iterator {
            all_combinations.push(combo);
        }

        let task_queue = Arc::new(ArrayQueue::new(all_combinations.len()));
        task_queue.push(all_combinations).unwrap();

        let results = if cuda_available {
            let (results, _) = process_gpu_combinations(
                &task_queue, numbers, params, &processed_count, progress_step, total_combinations
            );
            results
        } else {
            Vec::new()
        };

        return results;
    }

    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    #[derive(PartialEq, Eq)]
    struct HeapItem(Reverse<OrderedFloat<f64>>, OptimizationResult);

    impl PartialOrd for HeapItem {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    impl Ord for HeapItem {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.cmp(&other.0)
        }
    }

    let max_results: usize = params.max_results.parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData,
                                    format!("Ошибка парсинга max_results: {}", e)))?.unwrap_or(10000);
    let mut best_results = BinaryHeap::with_capacity(max_results);


    let mut current_batch = Vec::with_capacity(max_batch_size);
    let mut batch_counter = 0;


    let shared_task_queue = Arc::new(ArrayQueue::new(100)); // Храним до 100 батчей в очереди


    let task_queue_cpu = Arc::clone(&shared_task_queue);
    let numbers_for_cpu = numbers.clone();
    let params_for_cpu = params.clone();
    let processed_cpu_clone = Arc::clone(&processed_count);

    let cpu_thread = std::thread::spawn(move || {
        let cpu_start = Instant::now();
        let core_ids = core_affinity::get_core_ids()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Не удалось получить core_ids"))?;
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
                println!(
                    "CPU прогресс: {}%",
                    ((current + chunk.len()) * 100) / total_combinations
                );
            }
        }

        (cpu_results, cpu_start.elapsed())
    });


    for combination in param_iterator {
        current_batch.push(combination);

        if current_batch.len() >= max_batch_size {

            save_intermediate_batch(&current_batch, batch_counter)?;


            while shared_task_queue.push(current_batch.clone()).is_err() {

                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            current_batch.clear();
            batch_counter += 1;
        }
    }


    if !current_batch.is_empty() {

        while shared_task_queue.push(current_batch.clone()).is_err() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        current_batch.clear();
    }


    let (cpu_results, cpu_time) = cpu_thread.join().unwrap();


    let gpu_results = if cuda_available {
        process_gpu_combinations(
            &shared_task_queue,
            numbers,
            params,
            &processed_count,
            progress_step,
            total_combinations
        )
    } else {
        (Vec::new(), std::time::Duration::new(0, 0))
    };

    let (gpu_results, gpu_time) = gpu_results;


    let mut all_results = Vec::new();
    all_results.extend(cpu_results);
    all_results.extend(gpu_results);

    all_results.sort_by(|a, b| b.balance.partial_cmp(&a.balance).unwrap());

    if all_results.len() > max_results {
        all_results.truncate(max_results);
    }

    println!("\nСтатистика выполнения:");
    println!("CPU время: {:?}", cpu_time);
    println!("GPU время: {:?}", gpu_time);
    println!(
        "Найдено стратегий: {}",
        all_results.len()
    );

    Ok(all_results)
}


fn process_cpu_combinations(
    combinations: &[ParameterCombination],
    numbers: &Array1<f64>,
    params: &Params,
) -> Vec<OptimizationResult> {

    let core_ids = core_affinity::get_core_ids().unwrap();

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
        rustacuda::init(CudaFlags::empty())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Ошибка инициализации CUDA: {}", e)))?;
        let device = Device::get_device(0)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Ошибка получения CUDA устройства: {}", e)))?;
        let _context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        ).unwrap();

        let ptx = CString::new(include_str!("../cuda/kernel.ptx")).unwrap();
        let module = Module::load_from_string(&ptx)
            .map_err(|e| to_io_error(e, "Ошибка загрузки CUDA модуля"))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let numbers_slice = numbers.as_slice()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Не удалось получить slice из numbers"))?;
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

    let initial_bet = if params.bet_type == "fixed" {
        round_to_cents(params.stake)
    } else {
        round_to_cents(params.initial_balance * (combo.stake_percent / 100.0))
    };

    let mut test_balance = round_to_cents(params.initial_balance);
    let mut test_stake = initial_bet;
    for _ in 0..combo.attempts {
        if test_stake > test_balance {

            return None;
        }
        test_balance = round_to_cents(test_balance - test_stake);
        test_stake = round_to_cents(test_stake * combo.multiplier);
    }

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

fn save_intermediate_batch(batch: &[ParameterCombination], batch_id: usize) -> std::io::Result<()> {

    let temp_dir = Path::new("temp_batches");
    if !temp_dir.exists() {
        create_dir_all(temp_dir)?;
    }

    let file_path = temp_dir.join(format!("batch_{}.bin", batch_id));
    let file = File::create(file_path)?;
    let writer = BufWriter::new(file);

    bincode::serialize_into(writer, batch)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

fn load_intermediate_batch(batch_id: usize) -> std::io::Result<Vec<ParameterCombination>> {
    let file_path = Path::new("temp_batches").join(format!("batch_{}.bin", batch_id));
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    bincode::deserialize_from(reader)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

fn save_best_results(results: &[OptimizationResult], batch_id: usize) -> std::io::Result<()> {
    let temp_dir = Path::new("temp_results");
    if !temp_dir.exists() {
        create_dir_all(temp_dir)?;
    }

    let file_path = temp_dir.join(format!("results_{}.bin", batch_id));
    let file = File::create(file_path)?;
    let writer = BufWriter::new(file);

    bincode::serialize_into(writer, results)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}
fn to_io_error<E: std::error::Error>(e: E, message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("{}: {}", message, e))
}

// Использование:
