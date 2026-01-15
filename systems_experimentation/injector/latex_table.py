import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats

CONFIDENCE_LEVEL = 0.95
STRATEGIES = [
    "Full-Stream Filtering", "Persistence-Path Control",
    "Persistence-Path + VR", "Fixed-Rate Filtering", "Periodic Batching"
]


def calculate_ci(series):
    series = series.dropna()
    n = len(series)
    if n < 2: return 0
    std_err = stats.sem(series)
    return stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, n - 1) * std_err


def format_cell(mean, ci, precision=2):
    if pd.isna(mean) or mean == 0:
        return "--"
    return f"{mean:.{precision}f} \\tiny{{$\\pm${ci:.{precision}f}}}"


def parse_metadata(base_name):
    """Maps filename prefixes to formal strategy names."""
    strategy_map = {
        'filtered': "Full-Stream Filtering",
        'fullstream': "Full-Stream Filtering",
        'filtered': "Persistence-Path",
        'fixed': "Fixed-Rate Filtering",
        'batched': "Periodic Batching",
        'unfiltered': "Unfiltered Baseline",
    }

    prefix = base_name.split('_')[0]
    strategy = strategy_map.get(prefix, "Unknown")

    threshold = np.nan
    match = re.search(r'_T(\d+)_(\d+)', base_name)
    if match:
        threshold = float(f"{match.group(1)}.{match.group(2)}")

    if strategy == 'Fixed-Rate Filtering':
        r_match = re.search(r'_R(\d+\.\d+)', base_name)
        strategy += r_match.group(1)

    if strategy == 'Periodic Batching':
        b_match = re.search(r'_B(\d+)', base_name)
        if b_match:
            strategy = f"{strategy} (B={b_match.group(1)})"

    return strategy, threshold


def parse_iostat(file_path):
    """Extracts average disk utilization from iostat.log."""
    if not os.path.exists(file_path): return 0
    utils = []
    wkb_per_s = []
    with open(file_path, 'r') as f:
        for line in f:
            if "nvme0n1" in line or "sda" in line:
                parts = line.split()
                if len(parts) > 1: utils.append(float(parts[-1]))
                if len(parts) > 1: wkb_per_s.append(float(parts[8]))
    if utils:
        return np.mean(utils), np.mean(wkb_per_s)
    return 0, 0


def extract_write_amplification(stats: str):
    for line in stats.splitlines():
        print(line)
        if line.strip().startswith("Sum"):
            parts = line.split()
            print(parts)
            # W-Amp is the 11th column in RocksDB compaction stats
            return float(parts[11])
    return None


def parse_db_stats_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 1. Fetch the direct keys
    metrics = {
        'l0_compression_ratio': float(data.get('waf_l0_ratio', 0)),
        'memtable_mb': data.get('memtable_size_bytes', 0) / (1024 * 1024),
        'sst_size_mb': data.get('total_sst_size_bytes', 0) / (1024 * 1024)
    }

    raw_text = data.get('raw_rocksdb_stats', "")

    metrics['total_w_amp'] = extract_write_amplification(raw_text)

    # 3. Extract Space Amplification
    space_amp_match = re.search(r"space amp: ([\d\.]+)", raw_text)
    metrics['space_amp'] = float(
        space_amp_match.group(1)) if space_amp_match else 0.0

    # 4. Extract Cumulative Ingest and Write Rate
    ingest_match = re.search(
        r"Cumulative writes:.*ingest: ([\d\.]+) GB, ([\d\.]+) MB/s", raw_text)
    if ingest_match:
        metrics['cumulative_ingest_gb'] = float(ingest_match.group(1))
        metrics['cumulative_write_mbs'] = float(ingest_match.group(2))

    # 5. Extract WAL Write stats
    wal_match = re.search(
        r"Cumulative WAL:.*written: ([\d\.]+) GB, ([\d\.]+) MB/s", raw_text)
    if wal_match:
        metrics['wal_written_gb'] = float(wal_match.group(1))
        metrics['wal_write_mbs'] = float(wal_match.group(2))

    # 6. Extract Write Stall Percentage
    stall_match = re.search(r"Cumulative stall: .*?, ([\d\.]+) percent",
                            raw_text)
    metrics['db_stall_pct'] = float(
        stall_match.group(1)) if stall_match else 0.0

    # 7. Extract Cache Usage
    cache_match = re.search(r"capacity: ([\d\.]+) MB.*?usage: ([\d\.]+) MB",
                            raw_text)
    if cache_match:
        metrics['block_cache_capacity_mb'] = float(cache_match.group(1))
        metrics['block_cache_usage_mb'] = float(cache_match.group(2))

    return metrics


def load_data(results_dir, anchor_file_pattern='metrics.json'):
    all_results = []
    anchor_files = glob.glob(
        os.path.join(results_dir, f'*{anchor_file_pattern}'))

    for f in anchor_files:
        base_id = os.path.basename(f).replace(anchor_file_pattern, "")
        strategy, threshold = parse_metadata(base_id)

        db_file = os.path.join(results_dir, f"{base_id}_db_stats.json")
        io_file = os.path.join(results_dir, f"{base_id}_iostat.log")
        locust_file = os.path.join(results_dir, f"{base_id}_stats.csv")

        data_row = {
            'Strategy': strategy,
            'Lambda': threshold,
            'Events_s': 0,
            'Lat_Avg': 0,
            'Lat_p95': 0,
            'Lat_p99': 0,
            'Lat_p99.99': 0,
            'Min_Lat': 0,
            'Max_Lat': 0,
            'RMW': 0,
            'WAF': 0,
            'IOPS': 0,
            'Util': 0,
            'wkb_per_s': 0,
        }

        # --- Parse Primary Metrics ---
        try:
            if anchor_file_pattern == '_metrics.json':
                print('parsgni metrics')
                with open(f, 'r') as j:
                    d = json.load(j)
                    data_row['RMW'] = (1 - d['skip_percentage']) * 100
        except Exception as e:
            print(f'Failed to parse {f} with:\n{e}')

        # --- Parse DB Stats (RMW, WAF) ---
        if os.path.exists(db_file):
            print('db file exists')
            try:
                metrics = parse_db_stats_file(db_file)
                print(metrics)
                # data_row['RMW'] = metrics['rmw_percentage']
                # data_row['WAF'] = metrics['waf_l0']
                data_row['WAF'] = metrics['total_w_amp']
                data_row['L0_Comp_Ratio'] = metrics.get(
                    'l0_compression_ratio', 0.0)

                # 3. Store the internal database throughput
                data_row['DB_Ingest_GB'] = metrics.get('cumulative_ingest_gb',
                                                       0.0)
                data_row['DB_Write_MBs'] = metrics.get('cumulative_write_mbs',
                                                       0.0)

                # 4. Critical performance indicators
                data_row['Stall_Pct'] = metrics.get('db_stall_pct', 0.0)
                data_row['Cache_Usage_MB'] = metrics.get(
                    'block_cache_usage_mb', 0.0)
                data_row['memtable_mb'] = metrics['memtable_mb']
                data_row['wal_write_mbs'] = metrics['wal_write_mbs']

                print(
                    f"Successfully updated row for {strategy}. WAF: {data_row['WAF']}"
                )
            except Exception as e:
                print(f'Failed to parse {f} with:\n{e}')

        # --- Parse System Stats (IOPS, Util) ---
        if os.path.exists(io_file):
            data_row['Util'], data_row['wkb_per_s'] = parse_iostat(io_file)

        # --- Parse Locust CSV (Latencies) ---
        if os.path.exists(locust_file):
            try:
                ldf = pd.read_csv(locust_file)
                agg = ldf[ldf['Name'] == 'Aggregated']
                if not agg.empty:
                    data_row['Events_s'] = agg.iloc[0]['Requests/s']
                    data_row['Lat_Avg'] = agg.iloc[0]['Average Response Time']
                    data_row['Lat_p95'] = agg.iloc[0]['95%']
                    data_row['Lat_p99'] = agg.iloc[0]['99%']
                    data_row['Lat_p99.99'] = agg.iloc[0]['99.99%']
                    data_row['Min_Lat'] = agg.iloc[0]['Min Response Time']
                    data_row['Max_Lat'] = agg.iloc[0]['Max Response Time']
            except Exception as e:
                print(f'Failed to parse {f} with:\n{e}')

        all_results.append(data_row)

    return pd.DataFrame(all_results)


def summarize_results(df):
    """
    Returns a summarized dataframe with 'mean' and 'CI'
    for all metrics grouped by Strategy and Lambda.
    """
    metrics = [
        'RMW',
        'Events_s',
        'Lat_Avg',
        'Lat_p95',
        'Lat_p99',
        'Lat_p99.99',
        'WAF',
        'Util',
        'wkb_per_s',
        'Stall_Pct',
        'memtable_mb',
        'Cache_Usage_MB',
        'wal_write_mbs',
    ]

    def mean_with_ci(x):
        m = x.mean()
        ci = calculate_ci(x)
        return f"{m:.5f} (±{ci:.5f})"

    summary = df.groupby(['Strategy', 'Lambda'],
                         dropna=False)[metrics].agg(mean_with_ci)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Print LaTeX table to stdout.')
    parser.add_argument('--experiment_dir',
                        type=str,
                        required=True,
                        help='Directory containing JSON results')
    args = parser.parse_args()

    if not os.path.isdir(args.experiment_dir):
        print(f"Error: {args.experiment_dir} is not a valid directory.")
        return

    df = load_data(args.experiment_dir, anchor_file_pattern='_metrics.json')
    df = load_data(args.experiment_dir, anchor_file_pattern='_db_stats.json')
    print(df)
    summary_df = summarize_results(df)
    print(df)
    print(summary_df)


if __name__ == "__main__":
    main()
