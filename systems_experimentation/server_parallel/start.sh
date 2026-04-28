#!/bin/bash
# start.sh

# Default to 4 workers if the variable is not set
NUM_WORKERS=${NUM_WORKERS:-4}

echo "Starting $NUM_WORKERS independent worker processes..."

for ((i=0; i<NUM_WORKERS; i++)); do
    PORT=$((8001 + i))
    echo "Starting Shard $i on port $PORT..."
    # Launch each worker in the background
    SHARD_ID=$i gunicorn --workers 1 --bind 0.0.0.0:$PORT flask_server:app &
done

# Wait for all background processes to keep the container running
wait
