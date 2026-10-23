#!/usr/bin/env bash
set -euo pipefail

# Compile binary in release mode
echo "[Runner] Compiling open-loop benchmark binary in release mode..."
cargo build --release --bin open_loop_bench

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
TOTAL_REPS=4

# Target endpoints
ENDPOINTS=("unfiltered" "persistence_path")

# Velocity thresholds for persistence_path
# VELOCITY_THRESHOLDS=("1.0" "10.0" "5000" "50000")
VELOCITY_THRESHOLDS=("1.0" "10.0" "5000")

# Offered load sweep (operations per second)
# TARGET_RATES=(10000 20000 30000 35000 40000 45000 50000 60000 70000 80000)
# Refined sweep range for local testing
TARGET_RATES=(
    10000  20000  30000  40000   # Sub-saturation baseline (both flat < 1ms)
    45000  50000  55000          # Unfiltered knee transition & queue explosion (~47.8k limit)
    65000  75000  85000          # Mid-range (Unfiltered blown out, Persistence Path running smooth)
    95000  105000 115000         # Persistence Path saturation knee (~93k-94k limit)
)

# Worker thread concurrency setting
WORKER_THREADS=4

echo "================================================================="
echo " Starting Open-Loop Throughput/Latency Sweep Benchmark Suite"
echo " Target Directory    : ./open_loop_experiments/$TIMESTAMP/"
echo " Endpoints           : ${ENDPOINTS[*]}"
echo " Target Rates (ops)  : ${TARGET_RATES[*]}"
echo " Velocity Thresholds : ${VELOCITY_THRESHOLDS[*]}"
echo " Worker Threads      : $WORKER_THREADS"
echo " Repetitions         : $TOTAL_REPS"
echo "================================================================="

# 1. Unfiltered Baseline Sweep
echo ""
echo "=========================================="
echo " Benchmarking Open-Loop Baseline: unfiltered"
echo "=========================================="

for RATE in "${TARGET_RATES[@]}"; do
    for REP in $(seq 0 $((TOTAL_REPS - 1))); do
        echo "---> [unfiltered] Target Rate: $RATE ops/s | Trial $REP of $TOTAL_REPS"

        ./target/release/open_loop_bench "unfiltered" "$REP" "$TIMESTAMP" "1.0" "$WORKER_THREADS" "$RATE"

        sleep 1
    done
done

# 2. Persistence Path Control Sweep across Thresholds & Rates
for THRESHOLD in "${VELOCITY_THRESHOLDS[@]}"; do
    echo ""
    echo "=========================================="
    echo " Benchmarking Open-Loop: persistence_path (Threshold: $THRESHOLD)"
    echo "=========================================="

    for RATE in "${TARGET_RATES[@]}"; do
        for REP in $(seq 0 $((TOTAL_REPS - 1))); do
            echo "---> [persistence_path] Threshold: $THRESHOLD | Target Rate: $RATE ops/s | Trial $REP of $TOTAL_REPS"

            ./target/release/open_loop_bench "persistence_path" "$REP" "$TIMESTAMP" "$THRESHOLD" "$WORKER_THREADS" "$RATE"

            sleep 1
        done
    done
done

echo ""
echo "================================================================="
echo " Open-Loop Experiment Sweep Complete!"
echo " Results saved in: ./open_loop_experiments/$TIMESTAMP/"
echo "================================================================="
