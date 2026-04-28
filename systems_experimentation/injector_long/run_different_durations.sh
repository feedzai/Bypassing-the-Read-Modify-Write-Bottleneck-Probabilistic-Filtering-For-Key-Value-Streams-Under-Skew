#!/bin/bash

# --- CONFIGURATION ---
TARGET_HOST="http://10.0.2.7" # The Flask AWS Private IP
IMAGE_NAME="locust-injector"
REPETITIONS=4
LOGS_DIR="long_running_experiment_results_$(date +%Y%m%d_%H%M%S)"
# 1. Build the Injector Image once
docker build -t $IMAGE_NAME -f Dockerfile .

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "Docker build failed. Check your Dockerfile.injector and requirements."
    exit 1
fi

mkdir -p "$LOGS_DIR"
chmod -R 777 "$LOGS_DIR"

THRESHOLDS=(0.01 0.05 0.1 1.0)
run_experiment() {
    # Arguments: 1:endpoint, 2:param_json, 3:name, 4:rep, 5:run_time (e.g., 1200s)
    local endpoint=$1; local param_json=$2; local name=$3; local rep=$4; local duration=$5
    
    # Append duration to the base log directory name
    local current_log_dir="${LOGS_DIR}_${duration}"
    local out="${name}_rep${rep}"

    # Ensure the specific directory exists
    mkdir -p "$current_log_dir"

    echo ">>> Starting: $name | Rep: $rep | Duration: $duration"
    
    curl -v -X POST "${TARGET_HOST}/api/v1/reset" \
         -H "Content-Type: application/json" \
         -d "$param_json"
    sleep 2

    # Using the new directory for volume mapping and CSV output
    docker run --rm -v "$(pwd)/$current_log_dir:/app/logs" \
        -e LOAD_TEST_ENDPOINT=$endpoint \
        -e PYTHONUNBUFFERED=1 \
        $IMAGE_NAME -f locust.py --host=$TARGET_HOST --headless \
        --csv="logs/$out" --run-time "$duration"

    # Save metrics to the time-stamped directory
    curl -s -X GET "${TARGET_HOST}/api/v1/stats" > "${current_log_dir}/${out}_metrics.json"
}


# Running injector for 5 minutes
echo "#########################################################"
echo "## Running Filtered Experiments (/filtered)              ##"
echo "#########################################################"
for T in "${THRESHOLDS[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment "/api/v1/filtered" "{\"threshold_minutes\": $T}" "filtered_T${T//./_}" $i "300s"
    done
done

# Running for 10 minutes
for T in "${THRESHOLDS[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment "/api/v1/filtered" "{\"threshold_minutes\": $T}" "filtered_T${T//./_}" $i "600s"
    done
done

# Running for 50 minutes
for T in "${THRESHOLDS[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment "/api/v1/filtered" "{\"threshold_minutes\": $T}" "filtered_T${T//./_}" $i "3000s"
    done
done




