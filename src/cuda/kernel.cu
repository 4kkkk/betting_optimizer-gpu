#include <stdio.h>
#include <math.h>

#define SM_TARGET 86
#define THREADS_PER_BLOCK 256
#define MAX_SHARED_MEM_PER_BLOCK 49152
#define WARP_SIZE 32

// Функция для округления до двух знаков после запятой (до сотых - центов)
__device__ double round_to_cents(double value) {
    return round(value * 100.0) / 100.0;
}

struct GpuOptimizationResult {
    double balance;
    double max_balance;
    int total_bets;
    int total_series;
    int winning_series;
    double profit;
    double initial_balance;
};

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

    const int param_offset = idx * 7;
    const int num_low = __double2int_rd(params[param_offset]);
    const double search_threshold = params[param_offset + 1];
    const double high_threshold = params[param_offset + 2];
    const double payout_threshold = params[param_offset + 3];
    const double multiplier = params[param_offset + 4];
    const double stake_param = params[param_offset + 5];
    const int attempts = __double2int_rd(params[param_offset + 6]);

    // Округляем начальный баланс
    double balance = round_to_cents(results[idx].initial_balance);
    double max_balance = balance;
    int total_bets = 0;
    int total_series = 0;
    int winning_series = 0;
    int consecutive_losses = 0;
    int i = num_low;
    bool insufficient_balance_flag = false;

    while (i < numbers_len && !insufficient_balance_flag) {
        bool sequence_valid = true;

        for (int j = 0; j < num_low && sequence_valid; j++) {
            if (i <= j || numbers[i - j - 1] > search_threshold) {
                sequence_valid = false;
            }
        }

        if (sequence_valid) {
            int search_i = i;
            while (search_i < numbers_len && numbers[search_i] < high_threshold) {
                search_i++;
            }

            if (search_i < numbers_len && numbers[search_i] >= high_threshold) {
                total_series++;

                // Проверка достаточности баланса для всей серии ставок до начала
                double test_balance = balance;
                // Округляем начальную ставку
                double test_stake = bet_type == 0 ?
                    round_to_cents(base_stake) :
                    round_to_cents(balance * (stake_param / 100.0));

                for (int test_attempt = 0; test_attempt < attempts; test_attempt++) {
                    if (test_stake > test_balance) {
                        insufficient_balance_flag = true;
                        break; // Выход из цикла проверки
                    }
                    // Округляем вычитание ставки из тестового баланса
                    test_balance = round_to_cents(test_balance - test_stake);
                    // Округляем увеличение ставки
                    test_stake = round_to_cents(test_stake * multiplier);
                }

                if (insufficient_balance_flag) {
                    break; // Выход из основного цикла сразу
                }

                int betting_attempts = 0;
                // Округляем начальную ставку
                double current_stake = bet_type == 0 ?
                    round_to_cents(base_stake) :
                    round_to_cents(balance * (stake_param / 100.0));

                int current_i = search_i;

                while (betting_attempts <= attempts - 1 && current_i < numbers_len - 1) {
                    if (current_stake > balance) {
                        break;
                    }

                    current_i++;
                    total_bets++;
                    // Округляем вычитание ставки из баланса
                    balance = round_to_cents(balance - current_stake);

                    if (numbers[current_i] >= payout_threshold) {
                        // Округляем выигрыш
                        double win = round_to_cents(current_stake * payout_threshold);
                        // Округляем прибавление выигрыша к балансу
                        balance = round_to_cents(balance + win);
                        winning_series++;
                        consecutive_losses = 0;
                        max_balance = fmax(balance, max_balance);
                        break;
                    } else {
                        consecutive_losses++;
                        // Округляем увеличение ставки
                        current_stake = round_to_cents(current_stake * multiplier);
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

    if (insufficient_balance_flag) {
        results[idx].profit = -1.0;
    } else {
        // Округляем все финальные результаты
        results[idx].balance = round_to_cents(balance);
        results[idx].max_balance = round_to_cents(max_balance);
        results[idx].total_bets = total_bets;
        results[idx].total_series = total_series;
        results[idx].winning_series = winning_series;
        // Округляем прибыль
        results[idx].profit = round_to_cents(balance - results[idx].initial_balance);
    }
}