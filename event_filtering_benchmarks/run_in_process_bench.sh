#!/usr/bin/env bash
set -euo pipefail

# Build release binary
echo "[Runner] Compiling in-process benchmark binary in release mode..."
cargo build --release --bin in_process_bench

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
TOTAL_REPS=5

# UNTHRESHOLDED_ENDPOINTS=("unfiltered" "batched")
UNTHRESHOLDED_ENDPOINTS=("unfiltered")
# VELOCITY_FILTERED_ENDPOINTS=("persistence_path" "full_stream_filtering")
VELOCITY_FILTERED_ENDPOINTS=("persistence_path")

VELOCITY_THRESHOLDS=("1.0" "10.0" "5000" "50000")
#                         1     10    5000
FIXED_PROBABILITIES=("0.23" "0.40" "0.85" "1.0")
# THREAD_SWEEP=(1 2 4 8 16 32)
THREAD_SWEEP=(1)
# THREAD_SWEEP=(32)

echo "================================================================="
echo " Starting Full Multi-Thread Concurrency Sweep Benchmark Suite"
echo " Target Folder       : ./experiment_results/$TIMESTAMP/"
echo " Baselines           : ${UNTHRESHOLDED_ENDPOINTS[*]}"
echo " Velocity Filtered   : ${VELOCITY_FILTERED_ENDPOINTS[*]}"
echo " Velocity Thresholds : ${VELOCITY_THRESHOLDS[*]}"
echo " Fixed Rate Probs    : ${FIXED_PROBABILITIES[*]}"
echo " Thread Sweep        : ${THREAD_SWEEP[*]}"
echo " Repetitions         : $TOTAL_REPS"
echo "================================================================="

# 1. Baseline Endpoints (Unfiltered & Batched)
for ENDPOINT in "${UNTHRESHOLDED_ENDPOINTS[@]}"; do
    echo ""
    echo "=========================================="
    echo " Benchmarking Baseline Endpoint: $ENDPOINT"
    echo "=========================================="

    for THREADS in "${THREAD_SWEEP[@]}"; do
        for REP in $(seq 0 $((TOTAL_REPS - 1))); do
            echo ""
            echo "---> [$ENDPOINT] Threads: $THREADS | Trial $REP of $TOTAL_REPS"

            ./target/release/in_process_bench "$ENDPOINT" "$REP" "$TIMESTAMP" "1.0" "$THREADS"

            sleep 1
        done
    done
done

# 2. Velocity-Adaptive Filtered Endpoints (Persistence Path & Full Stream Filtering)
for ENDPOINT in "${VELOCITY_FILTERED_ENDPOINTS[@]}"; do
    for THRESHOLD in "${VELOCITY_THRESHOLDS[@]}"; do
        echo ""
        echo "=========================================="
        echo " Benchmarking Velocity Endpoint: $ENDPOINT (Threshold: $THRESHOLD)"
        echo "=========================================="

        for THREADS in "${THREAD_SWEEP[@]}"; do
            for REP in $(seq 0 $((TOTAL_REPS - 1))); do
                echo ""
                echo "---> [$ENDPOINT] Threshold: $THRESHOLD | Threads: $THREADS | Trial $REP of $TOTAL_REPS"

                ./target/release/in_process_bench "$ENDPOINT" "$REP" "$TIMESTAMP" "$THRESHOLD" "$THREADS"

                sleep 1
            done
        done
    done
done

# 3. Fixed Rate Probabilistic Filtering (Static Sample Probabilities)
# echo ""
# echo "=========================================="
# echo " Benchmarking Endpoint: fixed_rate_filtering"
# echo "=========================================="
# 
# for PROB in "${FIXED_PROBABILITIES[@]}"; do
#     for THREADS in "${THREAD_SWEEP[@]}"; do
#         for REP in $(seq 0 $((TOTAL_REPS - 1))); do
#             echo ""
#             echo "---> [fixed_rate_filtering] Probability: $PROB | Threads: $THREADS | Trial $REP of $TOTAL_REPS"
# 
#             ./target/release/in_process_bench "fixed_rate_filtering" "$REP" "$TIMESTAMP" "$PROB" "$THREADS"
# 
#             sleep 1
#         done
#     done
# done

echo ""
echo "================================================================="
echo " Concurrency Sweep Complete!"
echo " Results stored in: ./experiment_results/$TIMESTAMP/"
echo "================================================================="
