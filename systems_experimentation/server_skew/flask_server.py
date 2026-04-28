"""
Testing the endpoints

curl -X POST http://10.0.2.7/api/v1/reset \
     -H "Content-Type: application/json" \
     -d '{"threshold_minutes": 5, "fixed_filtering_rate": 1.0, "batched_n_updates": 1}'

curl -X POST http://10.0.2.7/api/v1/filtered \
     -H "Content-Type: application/json" \
     -d '{"merchant_id": "m_1", "unix_seconds": 1704067201}'


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

FILTERED_PATH = './filtered'
FILTERED_DB = Rdict(FILTERED_PATH)

import re
from typing import Optional


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
    global FILTERED_DB

    opts = get_low_mem_options()
    if FILTERED_DB: FILTERED_DB.close()
    cleanup_kv_stores(FILTERED_PATH)
    FILTERED_DB = Rdict(FILTERED_PATH, options=opts)


# Initial start
init_dbs()


@app.route('/api/v1/reset', methods=['POST'])
def reset_experiment():
    global CURRENT_THRESHOLD
    with db_lock:
        data = request.json or {}
        new_threshold_mins = data.get('threshold_minutes', 0)
        CURRENT_THRESHOLD = float(new_threshold_mins) * 60 * 24 * 7 * 4

        app.config['n_updates_filtered'] = 0
        app.config['n_skips_filtered'] = 0

        init_dbs()

        return jsonify({
            'status': 'reset_successful',
            'new_threshold_seconds': CURRENT_THRESHOLD,
            'fixed_filtering_rate': FIXED_FILTERING_RATE,
        }), 200


@app.route('/api/v1/filtered', methods=['POST'])
def handle_flitered():
    with db_lock:
        data = request.json
        id_ = data['merchant_id']
        t = data['unix_seconds']

        data = FILTERED_DB.get(id_)

        if data is None:
            intensity, profiles = (temporal_aggregations.DecayedSum(
                60 * 60 * 24 * 7 * 4), [
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        required=True,
    )
    args = parser.parse_args()
    app.run(port=args.port, debug=False)
