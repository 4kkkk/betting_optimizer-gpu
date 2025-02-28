#include <stdio.h>
#include <math.h>

#define SM_TARGET 86

// Константы для настройки производительности
#define MAX_SHARED_MEMORY_SIZE 49152  // Максимальный размер shared memory для SM 8.6 (RTX 3060)
#define BLOCK_SIZE_DEFAULT 256        // Для регистров

// Результаты оптимизации
struct GpuOptimizationResult {
    double balance;
    double max_balance;
    int total_bets;
    int total_series;
    int winning_series;
    double profit;
    double initial_balance;
};

// Оптимизированное ядро CUDA для RTX 3060
extern "C" __global__ void optimize_kernel(
    const double* __restrict__ numbers,
    const double* __restrict__ params,
    GpuOptimizationResult* __restrict__ results,
    const int numbers_len,
    const int params_len,
    const int bet_type,
    const double base_stake
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params_len) return;

    // Используем быстрый доступ к параметрам через кэшированные регистры
    const int param_offset = idx * 7;
    const int num_low = __double2int_rd(params[param_offset]);
    const double search_threshold = params[param_offset + 1];
    const double high_threshold = params[param_offset + 2];
    const double payout_threshold = params[param_offset + 3];
    const double multiplier = params[param_offset + 4];
    const double stake_param = params[param_offset + 5];
    const int attempts = __double2int_rd(params[param_offset + 6]);

    // Используем регистры для часто используемых переменных
    double balance = results[idx].initial_balance;
    double max_balance = balance;
    int total_bets = 0;
    int total_series = 0;
    int winning_series = 0;
    int consecutive_losses = 0;
    bool insufficient_balance_flag = false;

    // Вместо константных индексов используем переменную i
    int i = num_low;

    // Оптимизация: предварительная проверка длины последовательности
    if (i >= numbers_len) {
        results[idx].profit = -1.0;
        return;
    }

    // Предварительный тест на возможность выполнить все ставки в серии
    double test_balance = balance;
    double test_stake = bet_type == 0 ? base_stake : balance * (stake_param / 100.0);

    for (int test_attempt = 0; test_attempt < attempts; test_attempt++) {
        if (test_stake > test_balance) {
            insufficient_balance_flag = true;
            break;
        }
        test_balance -= test_stake;
        test_stake *= multiplier;
    }

    if (insufficient_balance_flag) {
        results[idx].profit = -1.0;
        return;
    }

    // Основной цикл оптимизации
    while (i < numbers_len && !insufficient_balance_flag) {
        bool sequence_valid = true;

        // Быстрая проверка валидности последовательности
        for (int j = 0; j < num_low && sequence_valid; j++) {
            if (i <= j || numbers[i - j - 1] > search_threshold) {
                sequence_valid = false;
            }
        }

        if (sequence_valid) {
            // Поиск соответствующего высокого порога
            int search_i = i;
            while (search_i < numbers_len && numbers[search_i] < high_threshold) {
                search_i++;
            }

            if (search_i < numbers_len && numbers[search_i] >= high_threshold) {
                total_series++;

                // Проверка баланса для всей серии ставок
                test_balance = balance;
                test_stake = bet_type == 0 ? base_stake : balance * (stake_param / 100.0);

                for (int test_attempt = 0; test_attempt < attempts; test_attempt++) {
                    if (test_stake > test_balance) {
                        insufficient_balance_flag = true;
                        break;
                    }
                    test_balance -= test_stake;
                    test_stake *= multiplier;
                }

                if (insufficient_balance_flag) {
                    break;
                }

                // Размещение ставок
                int betting_attempts = 0;
                double current_stake = bet_type == 0 ? base_stake : balance * (stake_param / 100.0);
                int current_i = search_i;

                while (betting_attempts <= attempts - 1 && current_i < numbers_len - 1) {
                    if (current_stake > balance) {
                        break;
                    }

                    current_i++;
                    total_bets++;
                    balance -= current_stake;

                    if (numbers[current_i] >= payout_threshold) {
                        balance += current_stake * payout_threshold;
                        winning_series++;
                        consecutive_losses = 0;
                        max_balance = fmax(balance, max_balance);
                        break;
                    } else {
                        consecutive_losses++;
                        current_stake *= multiplier;
                        betting_attempts++;
                    }
                }

                if (betting_attempts >= attempts) {
                    consecutive_losses = 0;
                }
                i = current_i + 1;
                continue;
            }
        }
        i++;
    }

    // Запись результатов
    if (insufficient_balance_flag) {
        results[idx].profit = -1.0;
    } else {
        results[idx].balance = balance;
        results[idx].max_balance = max_balance;
        results[idx].total_bets = total_bets;
        results[idx].total_series = total_series;
        results[idx].winning_series = winning_series;
        results[idx].profit = balance - results[idx].initial_balance;
    }
}