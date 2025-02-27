#include <stdio.h>
#include <math.h>

#define SM_TARGET 86

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

    double balance = results[idx].initial_balance;
    double max_balance = balance;
    int total_bets = 0;
    int total_series = 0;
    int winning_series = 0;
    int consecutive_losses = 0;
    int i = num_low;

    while (i < numbers_len) {
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
                int betting_attempts = 0;
                double current_stake = bet_type == 0 ? base_stake : balance * (stake_param / 100.0);
                int current_i = search_i;
                while (betting_attempts <= attempts - 1 && current_i < numbers_len - 1) {
                    // Проверка достаточности баланса для текущей ставки
                    if (current_stake > balance) {
                        // Если баланса не хватает, прерываем серию ставок
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
                    }                }

                if (betting_attempts >= attempts) {
                    consecutive_losses = 0;
                }
                i = current_i + 1;
                continue;
            }
        }
        i++;
    }

    results[idx].balance = balance;
    results[idx].max_balance = max_balance;
    results[idx].total_bets = total_bets;
    results[idx].total_series = total_series;
    results[idx].winning_series = winning_series;
    results[idx].profit = balance - results[idx].initial_balance;
}
