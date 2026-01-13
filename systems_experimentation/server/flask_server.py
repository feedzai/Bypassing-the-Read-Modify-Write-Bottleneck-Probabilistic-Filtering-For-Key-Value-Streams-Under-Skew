"""
Testing the endpoints

curl -X POST http://10.0.2.7/api/v1/reset \
     -H "Content-Type: application/json" \
     -d '{"threshold_minutes": 5, "fixed_filtering_rate": 1.0, "batched_n_updates": 1}'

curl -X POST http://10.0.2.7/api/v1/unfiltered \
     -H "Content-Type: application/json" \
     -d '{"merchant_id": "m_1", "unix_seconds": 1704067200}'

curl -X POST http://10.0.2.7/api/v1/filtered \
     -H "Content-Type: application/json" \
     -d '{"merchant_id": "m_1", "unix_seconds": 1704067201}'

curl -X POST http://10.0.2.7/api/v1/fixed_rate \
     -H "Content-Type: application/json" \
     -d '{"merchant_id": "m_1", "unix_seconds": 1704067202}'

curl -X POST http://10.0.2.7/api/v1/batched \
     -H "Content-Type: application/json" \
     -d '{"merchant_id": "m_1", "unix_seconds": 1704067203}'

curl -X POST http://10.0.2.7/api/v1/full_stream \
     -H "Content-Type: application/json" \
     -d '{
        "merchant_id": "merchant_123",
        "unix_seconds": 1704153600,
        "amount_usd": 100.0
     }'

curl -X POST http://10.0.2.7/api/v1/variance_reduced \
     -H "Content-Type: application/json" \
     -d '{
        "merchant_id": "m_1",
        "unix_seconds": 1704153600,
        "amount_usd": 5000.0
     }'

curl -X GET "http://10.0.2.7/api/v1/db-stats?db=batched"


#### LOCALLY #####

curl -X POST http://127.0.0.1:8000/api/v1/reset \
     -H "Content-Type: application/json" \
     -d '{"threshold_minutes": 5, "fixed_filtering_rate": 1.0, "batched_n_updates": 1}'

curl -X POST http://127.0.0.1:8000/api/v1/filtered \
     -H "Content-Type: application/json" \
     -d '{"merchant_id": "m_1", "unix_seconds": 1704067201}'

curl -X GET "http://127.0.0.1:8000/api/v1/db-stats?db=filtered"
"""
import re
import argparse
import os
import shutil
import pickle
import threading

import numpy as np
import math

from flask import Flask, request, jsonify

import temporal_aggregations
from rocksdict import Rdict, WriteOptions, WriteBatch, Options, BlockBasedOptions
import time

SYNC_OPTS = WriteOptions()
SYNC_OPTS.sync = True

db_lock = threading.Lock()


def get_low_mem_options():
    opts = Options()

    opts.set_write_buffer_size(2 * 1024 * 1024)
    opts.set_max_write_buffer_number(2)
    opts.set_target_file_size_base(2 * 1024 * 1024)

    return opts


def cleanup_kv_stores(path):
    """Cleans the KV-store contents without removing the mount point."""
    if os.path.exists(path):
        print(f'Cleaning contents of {path}')
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


HALF_LIFES = [
    60, 60 * 60, 60 * 60 * 24, 60 * 60 * 24 * 30, 60 * 60 * 24 * 30 * 2,
    60 * 60 * 24 * 30 * 3
]
CURRENT_THRESHOLD = None
FIXED_FILTERING_RATE = 1

app = Flask(__name__)

app.config['n_updates_filtered'] = 0
app.config['n_skips_filtered'] = 0

UNFILTERED_PATH = './unfiltered'
UNFILTERED_DB = Rdict(UNFILTERED_PATH)

FILTERED_PATH = './filtered'
FILTERED_DB = Rdict(FILTERED_PATH)

FIXED_FILTERING_PATH = './fixed_filtering'
FIXED_FILTERING_DB = Rdict(FIXED_FILTERING_PATH)

BATCHED_PATH = './batched'
BATCHED_DB = Rdict(BATCHED_PATH)

BATCH_CACHE = {}
BATCHED_N_UPDATES = 100
current_batch_count = 0

FULL_STREAM_PATH = './full_stream'
FULL_STREAM_DB = Rdict(FULL_STREAM_PATH)
GLOBAL_INTENSITY = {}

VARIANCE_REDUCED_PATH = './variance_reduced'
VR_DB = Rdict(VARIANCE_REDUCED_PATH)


@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    n_updates = app.config['n_updates_filtered']
    n_skips = app.config['n_skips_filtered']
    total = n_updates + n_skips

    if total > 0:
        skip_percentage = n_skips / total
    else:
        skip_percentage = 0.0

    return jsonify({
        'n_updates': n_updates,
        'n_skips': n_skips,
        'skip_percentage': skip_percentage,
    }), 200


def init_dbs():
    global UNFILTERED_DB, FILTERED_DB, FIXED_FILTERING_DB, BATCHED_DB, FULL_STREAM_DB, VR_DB

    opts = get_low_mem_options()
    if UNFILTERED_DB: UNFILTERED_DB.close()
    if FILTERED_DB: FILTERED_DB.close()
    if FIXED_FILTERING_DB: FIXED_FILTERING_DB.close()
    if BATCHED_DB: BATCHED_DB.close()
    if FULL_STREAM_DB: FULL_STREAM_DB.close()
    if VR_DB: VR_DB.close()

    cleanup_kv_stores(UNFILTERED_PATH)
    cleanup_kv_stores(FILTERED_PATH)
    cleanup_kv_stores(FIXED_FILTERING_PATH)
    cleanup_kv_stores(BATCHED_PATH)
    cleanup_kv_stores(FULL_STREAM_PATH)
    cleanup_kv_stores(VARIANCE_REDUCED_PATH)

    UNFILTERED_DB = Rdict(UNFILTERED_PATH, options=opts)
    FILTERED_DB = Rdict(FILTERED_PATH, options=opts)
    FIXED_FILTERING_DB = Rdict(FIXED_FILTERING_PATH, options=opts)
    BATCHED_DB = Rdict(BATCHED_PATH, options=opts)
    FULL_STREAM_DB = Rdict(FULL_STREAM_PATH, options=opts)
    VR_DB = Rdict(VARIANCE_REDUCED_PATH, options=opts)


# Initial start
init_dbs()


@app.route('/api/v1/reset', methods=['POST'])
def reset_experiment():
    global CURRENT_THRESHOLD
    global FIXED_FILTERING_RATE
    global BATCHED_N_UPDATES
    global current_batch_count
    global BATCH_CACHE
    with db_lock:
        data = request.json or {}
        new_threshold_mins = data.get('threshold_minutes', 0)
        CURRENT_THRESHOLD = float(new_threshold_mins) * 60
        FIXED_FILTERING_RATE = float(data.get('fixed_filtering_rate', 1))
        BATCHED_N_UPDATES = float(data.get('batched_n_updates', 1))
        current_batch_count = 0
        BATCH_CACHE = {}

        app.config['n_updates_filtered'] = 0
        app.config['n_skips_filtered'] = 0

        init_dbs()

        return jsonify({
            'status': 'reset_successful',
            'new_threshold_seconds': CURRENT_THRESHOLD,
            'fixed_filtering_rate': FIXED_FILTERING_RATE,
        }), 200


@app.route('/api/v1/unfiltered', methods=['POST'])
def handle_unfiltered():
    with db_lock:
        data = request.json
        id_ = data['merchant_id']
        t = data['unix_seconds']

        profiles = UNFILTERED_DB.get(id_)

        if profiles is None:
            profiles = [
                temporal_aggregations.DecayedSum(hf) for hf in HALF_LIFES
            ]
        else:
            profiles = pickle.loads(profiles)
        response = [c.update(t, 1, 1) for c in profiles]

        UNFILTERED_DB.put(id_, pickle.dumps(profiles), write_opt=SYNC_OPTS)

        return jsonify({'porfiles': response}), 200


@app.route('/api/v1/filtered', methods=['POST'])
def handle_flitered():
    with db_lock:
        data = request.json
        id_ = data['merchant_id']
        t = data['unix_seconds']

        data = FILTERED_DB.get(id_)

        if data is None:
            intensity, profiles = (temporal_aggregations.DecayedSum(60 * 60), [
                temporal_aggregations.DecayedSum(hf) for hf in HALF_LIFES
            ])
        else:
            intensity, profiles = pickle.loads(data)

        p = min(1, CURRENT_THRESHOLD / (intensity.get(t) + 1))

        response = [c.get(t) + 1 for c in profiles]
        if np.random.rand() < p:
            for c in profiles:
                c.update(t, 1, 1 / p)
            intensity.update(t, 1, 1 / p)
            FILTERED_DB.put(id_,
                            pickle.dumps((intensity, profiles)),
                            write_opt=SYNC_OPTS)
            app.config['n_updates_filtered'] += 1
        else:
            app.config['n_skips_filtered'] += 1
        return jsonify({'porfiles': response}), 200


@app.route('/api/v1/fixed_rate', methods=['POST'])
def handle_fixed_rate():
    with db_lock:
        data = request.json
        id_ = data['merchant_id']
        t = data['unix_seconds']

        profiles = FIXED_FILTERING_DB.get(id_)

        if profiles is None:
            profiles = [
                temporal_aggregations.DecayedSum(hf) for hf in HALF_LIFES
            ]
        else:
            profiles = pickle.loads(profiles)
        response = [c.update(t, 1, 1) for c in profiles]

        if np.random.rand() < FIXED_FILTERING_RATE:
            FIXED_FILTERING_DB.put(id_,
                                   pickle.dumps(profiles),
                                   write_opt=SYNC_OPTS)
            app.config['n_updates_filtered'] += 1
        else:
            app.config['n_skips_filtered'] += 1

        return jsonify({'porfiles': response}), 200


@app.route('/api/v1/batched', methods=['POST'])
def handle_batched():
    global current_batch_count, BATCH_CACHE
    with db_lock:
        data = request.json
        id_ = data['merchant_id']
        t = data['unix_seconds']

        if id_ in BATCH_CACHE:
            profiles = BATCH_CACHE[id_]
        else:
            db_data = BATCHED_DB.get(id_)
            if db_data is None:
                profiles = [
                    temporal_aggregations.DecayedSum(hf) for hf in HALF_LIFES
                ]
            else:
                profiles = pickle.loads(db_data)

        response = [c.update(t, 1, 1) for c in profiles]

        BATCH_CACHE[id_] = profiles

        app.config['n_updates_filtered'] += 1
        current_batch_count += 1

        if current_batch_count >= BATCHED_N_UPDATES:
            batch = WriteBatch()
            for merchant_id, merchant_profiles in BATCH_CACHE.items():
                batch.put(merchant_id, pickle.dumps(merchant_profiles))

            BATCHED_DB.write(batch, write_opt=SYNC_OPTS)

            BATCH_CACHE = {}
            current_batch_count = 0
        return jsonify({'porfiles': response}), 200


@app.route('/api/v1/full_stream', methods=['POST'])
def handle_full_stream():
    global GLOBAL_INTENSITY
    with db_lock:
        data = request.json
        id_ = data['merchant_id']
        t = data['unix_seconds']

        # 1. Tier 1: Always update the in-memory shadow intensity (p=1)
        if id_ not in GLOBAL_INTENSITY:
            GLOBAL_INTENSITY[id_] = temporal_aggregations.DecayedSum(60 * 60)

        GLOBAL_INTENSITY[id_].update(t, 1, 1)
        current_rate = GLOBAL_INTENSITY[id_].get(t)

        p = min(1.0, CURRENT_THRESHOLD / (current_rate + 1.0))

        db_data = FULL_STREAM_DB.get(id_)
        if db_data is None:
            profiles = [
                temporal_aggregations.DecayedSum(hf) for hf in HALF_LIFES
            ]
        else:
            profiles = pickle.loads(db_data)

        response = [c.get(t) for c in profiles]

        if np.random.rand() < p:
            for c in profiles:
                c.update(t, 1, 1 / p)

            FULL_STREAM_DB.put(id_,
                               pickle.dumps(profiles),
                               write_opt=SYNC_OPTS)
            app.config['n_updates_filtered'] += 1
        else:
            app.config['n_skips_filtered'] += 1

        return jsonify({'profiles': response}), 200


@app.route('/api/v1/db-stats', methods=['GET'])
def get_db_stats():
    db_type = request.args.get('db')
    db_map = {
        "unfiltered": UNFILTERED_DB,
        "filtered": FILTERED_DB,
        "fixed": FIXED_FILTERING_DB,
        "batched": BATCHED_DB,
        "full_stream": FULL_STREAM_DB,
    }

    target_db = db_map.get(db_type)

    if target_db is None:
        return jsonify({"error": f"Invalid DB: {db_type}"}), 400

    target_db.flush(wait=True)
    target_db.compact_range(None, None)
    time.sleep(10)

    try:
        waf = extract_write_amplification(
            target_db.property_value("rocksdb.stats"))
    except:
        print('failed to get waf from file')
        waf = None
    try:
        stats = {
            "db_type":
            db_type,
            "raw_rocksdb_stats":
            str(target_db.property_value("rocksdb.stats")),
            "total_sst_size_bytes":
            target_db.property_int_value("rocksdb.total-sst-files-size"),
            "memtable_size_bytes":
            target_db.property_int_value("rocksdb.cur-size-all-mem-tables"),
            "waf_l0_ratio":
            str(target_db.property_value(
                "rocksdb.compression-ratio-at-level0")),
            "waf":
            waf
        }
        return jsonify(stats), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/variance_reduced', methods=['POST'])
def handle_variance_reduced():
    with db_lock:
        data = request.json

        merchant_id = data['merchant_id']
        t = data['unix_seconds']
        q = data['amount_usd']

        # 1. Retrieve current state
        db_data = VR_DB.get(merchant_id)
        if db_data is None:
            intensity = temporal_aggregations.DecayedSum(60 * 60)
            var = temporal_aggregations.DecayedVariance(60 * 60)
            mean = temporal_aggregations.DecayedMean(60 * 60)
            profiles = [
                temporal_aggregations.DecayedSum(hf) for hf in HALF_LIFES
            ]
        else:
            intensity, var, mean, profiles = pickle.loads(db_data)

        rate = intensity.get(t)
        p_base = min(1.0, CURRENT_THRESHOLD / (rate + 1e-7))

        v = var.get(t)
        mu = mean.get(t)

        var_alpha = app.config.get('VAR_ALPHA', 1.0)

        logit_p = temporal_aggregations.jit_logit(p_base)
        z_score = (q - mu) / (math.sqrt(v) + 1e-7)
        p_final = temporal_aggregations.jit_sigmoid(logit_p +
                                                    var_alpha * z_score)

        response_data = [c.get(t) for c in profiles]

        if np.random.rand() < p_final:
            for c in profiles:
                c.update(t, 1, p_final)

            intensity.update(t, 1, p_final)
            var.update(t, q, p_final)
            mean.update(t, q, p_final)

            # Persist to RocksDB
            VR_DB.put(merchant_id,
                      pickle.dumps((intensity, var, mean, profiles)),
                      write_opt=SYNC_OPTS)

            app.config['n_updates_filtered'] += 1
        else:
            app.config['n_skips_filtered'] += 1

        return jsonify({'profiles': response_data}), 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        required=True,
    )
    args = parser.parse_args()
    app.run(port=args.port, debug=False)
