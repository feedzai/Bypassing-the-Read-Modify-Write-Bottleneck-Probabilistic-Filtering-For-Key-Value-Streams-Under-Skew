#!/bin/bash

# --- CONFIGURATION ---
TARGET_HOST="http://10.0.2.7" # The Flask AWS Private IP
IMAGE_NAME="locust-injector"
DATA_PATH="$(pwd)/data.parquet"
LOGS_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"
REPETITIONS=30

# 1. Build the Injector Image once
docker build -t $IMAGE_NAME -f Dockerfile .

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "Docker build failed. Check your Dockerfile.injector and requirements."
    exit 1
fi

mkdir -p "$LOGS_DIR"
chmod -R 777 "$LOGS_DIR"

THRESHOLDS=(0.001 0.005 0.01 0.05 0.1 1.0)
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
        $IMAGE_NAME -f locust.py --host=$TARGET_HOST --headless \
        --csv="logs/$out" --run-time 1200s

    curl -s -X GET "${TARGET_HOST}/api/v1/stats" > "${LOGS_DIR}/${out}_metrics.json"
}

run_experiment_profiled() {
    local endpoint=$1
    local db_key=$2
    local param_json=$3
    local name=$4
    local rep=$5
    local out="${name}_rep${rep}"
    local server_ip="10.0.2.7"

    echo "-------------------------------------------------------"
    echo "PROFILING: $out | Target: $db_key"
    echo "-------------------------------------------------------"

    curl -v -X POST "${TARGET_HOST}/api/v1/reset" \
         -H "Content-Type: application/json" \
         -d "$param_json"
    sleep 2

    ssh -n -i .ssh/ai-research.pem ubuntu@$server_ip "iostat -dx 1 1200 > /tmp/${out}_iostat.log" &
    local IO_PID=$!

    docker run --rm \
        -e LOAD_TEST_ENDPOINT=$endpoint \
        -e PYTHONUNBUFFERED=1 \
        $IMAGE_NAME \
        -f /app/locust_const.py \
        --host=$TARGET_HOST \
        --headless \
        --run-time 1200s \
        --only-summary

    ssh -n -i .ssh/ai-research.pem ubuntu@$server_ip "pkill iostat"
    
    echo "Waiting 10s for background DB settling..."
    sleep 10

    echo "Downloading metrics..."
    scp -i .ssh/ai-research.pem "ubuntu@${server_ip}:/tmp/${out}_iostat.log" "${LOGS_DIR}/"
    curl -s -X GET "${TARGET_HOST}/api/v1/db-stats?db=${db_key}" > "${LOGS_DIR}/${out}_db_stats.json"
}
# --- EXECUTION ---

echo "#########################################################"
echo "## Running Unfiltered Baseline Experiments (/unfiltered) ##"
echo "#########################################################"
for i in $(seq 1 $REPETITIONS); do
    run_experiment "/api/v1/unfiltered" "{}" "unfiltered_baseline" $i
done

echo "#########################################################"
echo "## Running Full Stream Filtering   ##"
echo "#########################################################"

for T in "${THRESHOLDS[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        PARAMS="{\"threshold_minutes\": $T}"
        LABEL="fullstream_T${T//./_}"
        run_experiment "/api/v1/full_stream" "$PARAMS" "$LABEL" $i
    done
done

echo ""
echo "#########################################################"
echo "## Running Filtered Experiments (/filtered)              ##"
echo "#########################################################"

for T in "${THRESHOLDS[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment "/api/v1/filtered" "{\"threshold_minutes\": $T}" "filtered_T${T//./_}" $i
    done
done

echo "#########################################################"
echo "## Running fixed filtering baseline ##"
echo "#########################################################"
FIXED_FILTERING_RATES=(0.14 0.45)
for R in "${FIXED_FILTERING_RATES[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment "/api/v1/fixed_rate" "{\"fixed_filtering_rate\": $R}" "fixed_R$R" $i
    done
done

echo ""
echo "All experiment results are saved in: ${LOGS_DIR}"

echo "#########################################################"
echo "## Running batched baseline ##"
echo "#########################################################"
BATCH_SIZES=(25 100)
for B in "${BATCH_SIZES[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment "/api/v1/batched" "{\"batched_n_updates\": $B}" "batched_B$B" $i
    done
done


echo ""
echo "#########################################################"
echo "## Running experiments for system metrics"
echo "#########################################################"

echo "#########################################################"
echo "## Running Unfiltered Baseline Experiments (/unfiltered) ##"
echo "#########################################################"
for i in $(seq 1 $REPETITIONS); do
    run_experiment_profiled "/api/v1/unfiltered" "unfiltered" "{}" "unfiltered_baseline" $i
done

echo "#########################################################"
echo "## Running Full Stream Filtering   ##"
echo "#########################################################"

for T in "${THRESHOLDS[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        PARAMS="{\"threshold_minutes\": $T}"
        LABEL="fullstream_T${T//./_}"
        
        run_experiment_profiled "/api/v1/full_stream" "full_stream" "$PARAMS" "$LABEL" $i
    done
done

echo ""
echo "#########################################################"
echo "## Running Filtered Experiments (/filtered)              ##"
echo "#########################################################"
for T in "${THRESHOLDS[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment_profiled "/api/v1/filtered" "filtered" "{\"threshold_minutes\": $T}" "filtered_T${T//./_}" $i
    done
done

echo "#########################################################"
echo "## Running fixed filtering baseline ##"
echo "#########################################################"
FIXED_FILTERING_RATES=(0.14 0.45)
for R in "${FIXED_FILTERING_RATES[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment_profiled "/api/v1/fixed_rate" "fixed" "{\"fixed_filtering_rate\": $R}" "fixed_R$R" $i
    done
done


echo "#########################################################"
echo "## Running batched baseline ##"
echo "#########################################################"
BATCH_SIZES=(100)
for B in "${BATCH_SIZES[@]}"; do
    for i in $(seq 1 $REPETITIONS); do
        run_experiment_profiled "/api/v1/batched" "batched" "{\"batched_n_updates\": $B}" "batched_B$B" $i
    done
done

echo ""
echo "All experiment results are saved in: ${LOGS_DIR}"
