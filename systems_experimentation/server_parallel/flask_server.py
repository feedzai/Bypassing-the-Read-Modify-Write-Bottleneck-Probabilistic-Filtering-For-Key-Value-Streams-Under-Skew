import os
import shutil
import numpy as np
from flask import Flask, request, jsonify

import temporal_aggregations
from rocksdict import Rdict, WriteOptions, Options

app = Flask(__name__)

# --- Shard Identity & DB Config ---
# Read the shard ID provided by the start.sh script
SHARD_ID = int(os.environ.get('SHARD_ID', 0))

SYNC_OPTS = WriteOptions()
SYNC_OPTS.sync = True

SUM_CHUNK_SIZE = temporal_aggregations._SUM_STRUCT.size

HALF_LIFES = [
    60, 60 * 60, 60 * 60 * 24, 60 * 60 * 24 * 30, 60 * 60 * 24 * 30 * 2,
    60 * 60 * 24 * 30 * 3
]
CURRENT_THRESHOLD = 5 * 60

# Lock-free metric tracking
app.config['n_updates_filtered'] = []
app.config['n_skips_filtered'] = []

UNFILTERED_DB = None
FILTERED_DB = None


def get_low_mem_options():
    opts = Options()
    opts.set_write_buffer_size(2 * 1024 * 1024)
    opts.set_max_write_buffer_number(2)
    opts.set_target_file_size_base(2 * 1024 * 1024)
    opts.set_max_background_jobs(4)
    return opts


def cleanup_kv_stores(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                pass


def init_dbs():
    global UNFILTERED_DB, FILTERED_DB
    opts = get_low_mem_options()

    if UNFILTERED_DB: UNFILTERED_DB.close()
    if FILTERED_DB: FILTERED_DB.close()

    # Create the data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)

    # Add 'data/' to the paths so they save inside the persistent volume
    u_path = f'./data/unfiltered_{SHARD_ID}'
    f_path = f'./data/filtered_{SHARD_ID}'

    cleanup_kv_stores(u_path)
    cleanup_kv_stores(f_path)

    UNFILTERED_DB = Rdict(u_path, options=opts)
    FILTERED_DB = Rdict(f_path, options=opts)


init_dbs()


@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    # Note: This returns stats for THIS SHARD ONLY.
    n_updates = len(app.config['n_updates_filtered'])
    n_skips = len(app.config['n_skips_filtered'])
    total = n_updates + n_skips
    return jsonify({
        'shard_id': SHARD_ID,
        'n_updates': n_updates,
        'n_skips': n_skips,
        'skip_percentage': n_skips / total if total > 0 else 0.0
    }), 200


@app.route('/api/v1/reset', methods=['POST'])
def reset_experiment():
    global CURRENT_THRESHOLD
    data = request.json or {}
    CURRENT_THRESHOLD = float(data.get('threshold_minutes', 5)) * 60
    app.config['n_updates_filtered'] = []
    app.config['n_skips_filtered'] = []
    init_dbs()
    return jsonify({'status': 'reset_successful', 'shard': SHARD_ID}), 200


@app.route('/api/v1/unfiltered', methods=['POST'])
def handle_unfiltered():
    data = request.json
    id_ = data['merchant_id']
    t = data['unix_seconds']

    # No Python locks needed because this worker process is single-threaded
    profiles_bytes = UNFILTERED_DB.get(id_)
    if profiles_bytes is None:
        profiles = [temporal_aggregations.DecayedSum(hf) for hf in HALF_LIFES]
    else:
        profiles = [
            temporal_aggregations.DecayedSum.from_bytes(
                profiles_bytes[i:i + SUM_CHUNK_SIZE])
            for i in range(0, len(profiles_bytes), SUM_CHUNK_SIZE)
        ]

    response = [c.update(t, 1, 1) for c in profiles]
    raw_bytes = b''.join([p.to_bytes() for p in profiles])
    UNFILTERED_DB.put(id_, raw_bytes, write_opt=SYNC_OPTS)

    return jsonify({'profiles': response}), 200


@app.route('/api/v1/filtered', methods=['POST'])
def handle_filtered():
    data = request.json
    id_ = data['merchant_id']
    t = data['unix_seconds']

    # No Python locks needed
    db_data = FILTERED_DB.get(id_)
    if db_data is None:
        intensity = temporal_aggregations.DecayedSum(3600)
        profiles = [temporal_aggregations.DecayedSum(hf) for hf in HALF_LIFES]
    else:
        intensity = temporal_aggregations.DecayedSum.from_bytes(
            db_data[:SUM_CHUNK_SIZE])
        profiles = [
            temporal_aggregations.DecayedSum.from_bytes(
                db_data[i:i + SUM_CHUNK_SIZE])
            for i in range(SUM_CHUNK_SIZE, len(db_data), SUM_CHUNK_SIZE)
        ]

    p = min(1.0, CURRENT_THRESHOLD / (intensity.get(t) + 1.0))
    response = [c.get(t) + 1 for c in profiles]

    if np.random.rand() < p:
        for c in profiles:
            c.update(t, 1, 1 / p)
        intensity.update(t, 1, 1 / p)

        raw_bytes = intensity.to_bytes() + b''.join(
            [p.to_bytes() for p in profiles])
        FILTERED_DB.put(id_, raw_bytes, write_opt=SYNC_OPTS)
        app.config['n_updates_filtered'].append(1)
    else:
        app.config['n_skips_filtered'].append(1)

    return jsonify({'profiles': response}), 200
