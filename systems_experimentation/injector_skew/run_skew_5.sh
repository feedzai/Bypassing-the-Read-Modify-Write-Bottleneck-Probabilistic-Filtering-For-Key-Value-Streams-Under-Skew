#!/bin/bash

# --- CONFIGURATION ---
TARGET_HOST="http://10.0.2.7" # The Flask AWS Private IP
IMAGE_NAME="locust-injector"
REPETITIONS=4
LOGS_DIR="skew_10_experiment_results_$(date +%Y%m%d_%H%M%S)"
# 1. Build the Injector Image once
docker build -t $IMAGE_NAME -f Dockerfile .

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "Docker build failed. Check your Dockerfile.injector and requirements."
    exit 1
fi

mkdir -p "$LOGS_DIR"
chmod -R 777 "$LOGS_DIR"

THRESHOLDS=(0.00003 0.00005 0.0001 0.0005 0.001)
run_experiment() {
    local endpoint=$1; local param_json=$2; local name=$3; local rep=$4
    local out="${name}_rep${rep}"

    echo ">>> Starting: $name | Rep: $rep"
    
    curl -v -X POST "${TARGET_HOST}/api/v1/reset" \
         -H "Content-Type: application/json" \
         -d "$param_json"
    sleep 2

    docker run --rm -v "$(pwd)/$LOGS_DIR:/app/logs" \
        -e LOAD_TEST_ENDPOINT=$endpoint \
        -e PYTHONUNBUFFERED=1 \
        $IMAGE_NAME -f locust_5.py --host=$TARGET_HOST --headless \
        --csv="logs/$out" --run-time 1200s

    curl -s -X GET "${TARGET_HOST}/api/v1/stats" > "${LOGS_DIR}/${out}_metrics.json"
}


echo "#########################################################"
echo "## Running Filtered Experiments (/filtered)              ##"
echo "#########################################################"
for T in "${THRESHOLDS[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment "/api/v1/filtered" "{\"threshold_minutes\": $T}" "filtered_T${T//./_}" $i
    done
done

