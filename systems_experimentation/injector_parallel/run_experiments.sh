#!/bin/bash

# --- CONFIGURATION ---
TARGET_HOST="http://10.0.2.7" # The Flask AWS Private IP
# TARGET_HOST="http://127.0.0.1"
IMAGE_NAME="locust-injector"
DATA_PATH="$(pwd)/data.parquet"
LOGS_DIR="threaded_experiment_results_$(date +%Y%m%d_%H%M%S)"
REPETITIONS=4

# IMPORTANT: Match this to the number of workers you launched on the server!
NUM_WORKERS=<MATCH THIS TO THE NUMBER OF WORKERS IN THE SERVER>

# 1. Build the Injector Image once
docker build -t $IMAGE_NAME -f Dockerfile .

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "Docker build failed. Check your Dockerfile and requirements."
    exit 1
fi

mkdir -p "$LOGS_DIR"
chmod -R 777 "$LOGS_DIR"

THRESHOLDS=(0.001 0.005 0.01 0.05 0.1 1.0)

run_experiment() {
    local endpoint=$1; local param_json=$2; local name=$3; local rep=$4
    local out="${name}_rep${rep}"

    echo ">>> Starting: $name | Rep: $rep"
    
    # Reset all workers across their respective ports
    echo "Resetting $NUM_WORKERS workers..."
    for ((i=0; i<NUM_WORKERS; i++)); do
        PORT=$((8001 + i))
        curl -s -X POST "${TARGET_HOST}:${PORT}/api/v1/reset" \
             -H "Content-Type: application/json" \
             -d "$param_json" > /dev/null
    done
    sleep 2

    # Launch Locust with multi-processing enabled
    docker run --rm -v "$(pwd)/$LOGS_DIR:/app/logs" \
        -e LOAD_TEST_ENDPOINT=$endpoint \
        -e NUM_WORKERS=$NUM_WORKERS \
        -e PYTHONUNBUFFERED=1 \
        $IMAGE_NAME -f locust.py --host=$TARGET_HOST --headless --users 50 \
        --processes $NUM_WORKERS \
        --csv="logs/$out" --run-time 1200s

    # Gather metrics from all workers and save them
    echo "Gathering metrics..."
    for ((i=0; i<NUM_WORKERS; i++)); do
        PORT=$((8001 + i))
        curl -s -X GET "${TARGET_HOST}:${PORT}/api/v1/stats" > "${LOGS_DIR}/${out}_metrics_shard${i}.json"
    done
}

# echo "#########################################################"
# echo "## Running Unfiltered Baseline Experiments (/unfiltered) ##"
# echo "#########################################################"
# for i in $(seq 1 $REPETITIONS); do
#     run_experiment "/api/v1/unfiltered" "{}" "unfiltered_baseline" $i
# done
 
echo "#########################################################"
echo "## Running Filtered Experiments (/filtered)              ##"
echo "#########################################################"
for T in "${THRESHOLDS[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment "/api/v1/filtered" "{\"threshold_minutes\": $T}" "filtered_T${T//./_}" $i
    done
done
