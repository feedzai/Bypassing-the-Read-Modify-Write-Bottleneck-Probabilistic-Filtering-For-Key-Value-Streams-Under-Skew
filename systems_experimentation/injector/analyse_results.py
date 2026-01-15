import os
import json
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse  # Added for CLI support

import numpy as np
from scipy import stats

RESULTS_DIR_PATTERN = "experiment_results_*"
CSV_PATTERN = "*_stats.csv"

COLOR_BLUE = "#4B6C9B"  # Muted Steel Blue
COLOR_ORANGE = "#C1440E"  # Dark Orange
COLOR_FILL = "#E8C9A1"  # Sand/Tan for CI/Density
COLOR_GRID = "#444444"
COLOR_PURPLE = "#6A4C93"  # Complements the blue/orange palette

CONFIDENCE_LEVEL = 0.95

# Global plotting config
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif"],
    "text.usetex": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "axes.unicode_minus": False,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.titlepad": 22,
    "grid.alpha": 0.25,
    "axes.edgecolor": COLOR_GRID,
    "legend.fontsize": 20,
    "figure.dpi": 300,
})


def extract_threshold(run_type):
    """Extracts the numerical threshold from the RunType string."""
    match = re.search(r'\d+\.?\d*', run_type)
    return float(match.group(0)) if match else np.nan


def parse_filename(filename):
    base_name = os.path.basename(filename).replace('_stats.csv', '')
    # base_name = os.path.basename(base_name).replace('_final_metrics.json', '')
    base_name = os.path.basename(base_name).replace('_metrics.json', '')
    parts = base_name.split('_')
    endpoint = parts[0]

    if endpoint == 'filtered' and len(parts) > 1 and parts[1].startswith('T'):
        threshold_str = parts[1][1:] + '.' + parts[2]
        threshold = float(threshold_str)
        run_type = f"Filtered ($\\Lambda={threshold}$)"
    else:
        threshold = 100.0
        run_type = "Unfiltered Baseline"

    print(filename)
    print(parts)
    repetition = int(re.sub(r'\D', '', parts[-1]))
    return endpoint, threshold, run_type, repetition


def load_and_aggregate_json(results_dir):
    all_data = []
    all_json_ = glob.glob(os.path.join(results_dir, '*.json'))
    all_json = [f for f in all_json_ if 'filtered' in f]
    all_json += [f for f in all_json_ if 'unfiltered' in f]
    for json_file in all_json:
        with open(json_file, 'r') as f:
            data = json.load(f)
        endpoint, threshold, run_type, repetition = parse_filename(json_file)
        all_data.append({
            'Endpoint': endpoint,
            'Threshold': threshold,
            'RunType': run_type,
            'Repetition': repetition,
            'filtered_volume': float(data['skip_percentage']),
            'kept_volume': 1 - float(data['skip_percentage']),
        })
    return pd.DataFrame(all_data)


def load_and_aggregate_data(results_dir):
    all_data = []
    all_csvs_ = glob.glob(os.path.join(results_dir, CSV_PATTERN))
    all_csvs = [f for f in all_csvs_ if 'filtered' in f]
    all_csvs += [f for f in all_csvs_ if 'unfiltered' in f]
    for csv_file in all_csvs:
        try:
            df = pd.read_csv(csv_file)
            total_row = df.iloc[-1]
            endpoint, threshold, run_type, repetition = parse_filename(
                csv_file)
            record = {
                'Endpoint': endpoint,
                'Threshold': threshold,
                'RunType': run_type,
                'Repetition': repetition,
                'RPS': total_row['Requests/s'],
                'AvgLatency': total_row['Average Response Time'],
                'P95Latency': total_row['95%'],
            }
            all_data.append(record)
        except Exception as e:
            print(f"Skipping file {csv_file} due to error: {e}")
            continue
    if not all_data:
        raise ValueError("No valid data rows found in CSV files.")
    return pd.DataFrame(all_data)


def calculate_confidence_interval(series):
    n = len(series)
    if n < 2: return 0
    std_err = stats.sem(series)
    return 1.96 * std_err


def create_summary_dataframe(df, metric_keys):
    df['SortKey'] = df.apply(lambda row: row['Threshold']
                             if row['Endpoint'] == 'filtered' else 1000,
                             axis=1)
    df.sort_values(by='SortKey', ascending=True, inplace=True)
    grouped = df.groupby('RunType', sort=False)
    summary = grouped[metric_keys].agg(['mean', 'std', 'count']).reset_index()
    summary.columns = [
        '_'.join(col).strip() if col[1] else col[0]
        for col in summary.columns.values
    ]
    summary.rename(columns={'RunType_': 'RunType'}, inplace=True)
    for metric in metric_keys:
        summary[f'{metric}_CI'] = summary.apply(
            lambda row: calculate_confidence_interval(df[df['RunType'] == row[
                'RunType']][metric]),
            axis=1)
    summary['Endpoint'] = summary['RunType'].apply(
        lambda x: 'unfiltered' if 'Unfiltered' in x else 'filtered')
    summary['Threshold'] = summary['RunType'].apply(extract_threshold)
    return summary


def plot_tradeoff_curve(summary_df, output_dir, filename, ci_hatch="////"):
    baseline_df = summary_df[summary_df['Endpoint'] == 'unfiltered'].iloc[0]
    filtered_df = summary_df[summary_df['Endpoint'] == 'filtered'].copy()
    thresholds = filtered_df['Threshold'].tolist()
    _, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()
    metrics_to_plot = [('RPS', 'Throughput (Requests/s)', 'darkgreen', 's',
                        ax),
                       ('P95Latency', 'P95 Latency (ms)', 'darkred', '^', ax2)]

    for idx, (metric, label, color, marker,
              plot_ax) in enumerate(metrics_to_plot):
        mean_col, ci_col = f'{metric}_mean', f'{metric}_CI'
        means, ci = filtered_df[mean_col].tolist(), filtered_df[ci_col].tolist(
        )
        plot_ax.fill_between(thresholds,
                             np.array(means) - np.array(ci),
                             np.array(means) + np.array(ci),
                             color=color,
                             alpha=0.10,
                             hatch=ci_hatch,
                             edgecolor=color,
                             linewidth=0.0,
                             zorder=1.2)
        plot_ax.plot(thresholds,
                     means,
                     marker=marker,
                     linestyle="-",
                     color=color,
                     linewidth=1.8,
                     markersize=7,
                     label=label,
                     zorder=2.0)
        base_mean, base_ci = baseline_df[mean_col], baseline_df[ci_col]
        plot_ax.axhline(
            base_mean,
            linestyle="--",
            color=COLOR_ORANGE,
            linewidth=1.6,
            alpha=0.8,
            label=f'Unfiltered baseline ({metric})' if idx == 0 else None)
        plot_ax.fill_between(thresholds, (base_mean - base_ci),
                             (base_mean + base_ci),
                             color=COLOR_ORANGE,
                             alpha=0.05,
                             hatch='.',
                             edgecolor=color,
                             linewidth=0.0,
                             zorder=1.1)

    ax.set_xscale("log")
    ax.set_xlabel("Sampling threshold $\\Lambda$")
    ax.set_title("Performance Trade-off: Metrics vs. Sampling Threshold")
    ax.grid(axis='both', linestyle='--', alpha=0.6)
    ax.set_ylabel("Throughput (Requests/s)", color='darkgreen')
    ax.tick_params(axis='y', labelcolor='darkgreen')
    ax2.set_ylabel("95th Percentile Latency (ms)", color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def plot_metric_vs_volume_tradeoff(summary_df_perf,
                                   summary_df_volume,
                                   metrics_to_plot_config,
                                   output_dir,
                                   filename,
                                   ci_hatch="////"):
    filtered_perf = summary_df_perf[summary_df_perf['Endpoint'] ==
                                    'filtered'].copy()
    filtered_volume = summary_df_volume[summary_df_volume['Endpoint'] ==
                                        'filtered'].copy()
    if filtered_volume.empty: return
    tradeoff_df = pd.merge(
        filtered_perf,
        filtered_volume[['RunType', 'kept_volume_mean', 'kept_volume_CI']],
        on='RunType')
    baseline_perf = summary_df_perf[summary_df_perf['Endpoint'] ==
                                    'unfiltered'].iloc[0]
    BASELINE_VOLUME = 1.0
    _, ax = plt.subplots(figsize=(8, 6), layout='constrained')

    for idx, config in enumerate(metrics_to_plot_config):
        metric, color, marker = config['metric'], config['color'], config[
            'marker']
        mean_col, ci_col = f'{metric}_mean', f'{metric}_CI'
        means, ci = tradeoff_df[mean_col].tolist(), tradeoff_df[ci_col].tolist(
        )
        x_means, x_ci = tradeoff_df['kept_volume_mean'].tolist(
        ), tradeoff_df['kept_volume_CI'].tolist()

        ax.fill_between(x_means,
                        np.array(means) - np.array(ci),
                        np.array(means) + np.array(ci),
                        color=COLOR_BLUE,
                        alpha=0.10,
                        hatch=ci_hatch,
                        edgecolor=color,
                        linewidth=0.0,
                        zorder=1.2)
        ax.plot(x_means,
                means,
                marker=marker,
                linestyle="-",
                color=COLOR_BLUE,
                linewidth=1.8,
                markersize=7,
                label=config['label'],
                zorder=2.0)
        ax.errorbar(x_means,
                    means,
                    xerr=x_ci,
                    fmt='none',
                    capsize=5,
                    color=COLOR_BLUE,
                    linewidth=1.0,
                    alpha=0.5)

        base_mean, base_ci = baseline_perf[mean_col], baseline_perf[ci_col]
        ax.axvspan(BASELINE_VOLUME - x_ci[0],
                   BASELINE_VOLUME + x_ci[0],
                   color='lightgray',
                   alpha=0.2,
                   zorder=0.5)
        ax.axhline(base_mean,
                   linestyle="--",
                   color=COLOR_ORANGE,
                   linewidth=1.8,
                   alpha=0.9,
                   label='Unfiltered baseline' if idx == 0 else None,
                   zorder=2.5)
        ax.axhspan(base_mean - base_ci,
                   base_mean + base_ci,
                   color=COLOR_ORANGE,
                   alpha=0.15,
                   linewidth=0.0,
                   zorder=1.1)

    ax.set_xlabel("Average processed volume")
    ax.set_title(r"\textbf{Throughput at processed volume}")
    ax.grid(axis='both', linestyle='--', alpha=0.6)
    ax.set_xlim(tradeoff_df['kept_volume_mean'].min() * 0.9, 1.05)
    ax.set_ylabel(metrics_to_plot_config[0]['x_label'])
    ax.tick_params(axis='y')
    ax.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze throughput experiment results.')
    parser.add_argument('--experiment_dir',
                        type=str,
                        required=True,
                        help='Path to the experiment results directory')
    args = parser.parse_args()

    results_dir = args.experiment_dir
    output_plots_dir = os.path.join(results_dir, "analysis_plots")

    if not os.path.isdir(results_dir):
        print(f"Error: Target results directory '{results_dir}' not found.")
        return

    print(f"Analyzing results from: {results_dir}")
    os.makedirs(output_plots_dir, exist_ok=True)

    raw_df = load_and_aggregate_data(results_dir)
    summary_df = create_summary_dataframe(raw_df,
                                          ['RPS', 'AvgLatency', 'P95Latency'])
    print(summary_df)

    raw_volumes = load_and_aggregate_json(results_dir)
    volume_df = create_summary_dataframe(raw_volumes,
                                         ['filtered_volume', 'kept_volume'])
    print(volume_df)

    plot_tradeoff_curve(summary_df,
                        output_plots_dir,
                        filename='performance_tradeoff_vs_threshold.pdf')

    PLOTTING_CONFIG = [{
        'metric': 'RPS',
        'label': 'Filtered',
        'color': COLOR_BLUE,
        'marker': 's',
        'axis': 'left',
        'x_label': 'Throughput (Events/second)'
    }]

    plot_metric_vs_volume_tradeoff(
        summary_df,
        volume_df,
        PLOTTING_CONFIG,
        output_plots_dir,
        filename='throughput_vs_volume_tradeoff.pdf')


if __name__ == '__main__':
    main()
