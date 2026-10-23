#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# Colorblind-friendly palette
CB_BLUE, CB_ORANGE, CB_GREEN = "#0072B2", "#D55E00", "#009E73"
CB_PURPLE, CB_YELLOW = "#CC79A7", "#D4AC0D"

# Strategy Color Map (Fixed color per method)
METHOD_COLORS = {
    "unfiltered": CB_BLUE,
    "persistence_path": CB_ORANGE,
}

# Volume Line Style Map (Consistent across methods)
VOLUME_STYLES = {
    "23%": {
        "ls": ":",
        "marker": "^",
        "label": r"$\sim 23\%$ Vol ($\Lambda=1.0$)"
    },
    "40%": {
        "ls": "-.",
        "marker": "s",
        "label": r"$\sim 40\%$ Vol ($\Lambda=10.0$)"
    },
    "85%": {
        "ls": "--",
        "marker": "v",
        "label": r"$\sim 85\%$ Vol ($\Lambda=5000$)"
    },
    "100%": {
        "ls": "-",
        "marker": "o",
        "label": r"$100\%$ Vol ($\Lambda=50000$)"
    },
}

# Configure Matplotlib publication defaults
HAS_TEX = False
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
    })
    HAS_TEX = True
except Exception:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
    })

# -----------------------------------------------------------------------------
# 1. Folder & Filename Parsing Utilities
# -----------------------------------------------------------------------------


def parse_folder_name(folder_name: str) -> Tuple[str, str]:
    """Extracts strategy name and threshold string from folder name."""
    if folder_name == "unfiltered":
        return "unfiltered", "-"

    prefix = "persistence_path"
    if folder_name.startswith(f"{prefix}_"):
        threshold_str = folder_name[len(prefix) + 1:]
        return prefix, threshold_str

    parts = folder_name.rsplit("_", 1)
    if len(parts) == 2 and re.match(r"^\d+(\.\d+)?$", parts[1]):
        return parts[0], parts[1]

    return folder_name, "-"


def parse_filename_metadata(filename: str) -> Optional[Tuple[int, int, float]]:
    """Extracts repetition ID, thread count, and rate from CSV filename."""
    pattern = r".*_rep_(\d+)_t(\d+)_rate(\d+)\.csv$"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2)), float(match.group(3))
    return None


def get_volume_key(strategy: str, threshold_str: str) -> str:
    """Maps (strategy, threshold) pair to standard volume percentage bucket."""
    if strategy == "unfiltered":
        return "100%"

    if threshold_str in ("1.0", "1", "1.00"):
        return "23%"
    elif threshold_str in ("10.0", "10", "10.00"):
        return "40%"
    elif threshold_str in ("5000", "5000.0"):
        return "85%"
    elif threshold_str in ("50000", "50000.0"):
        return "100%"

    return "100%"


# -----------------------------------------------------------------------------
# 2. CSV File Parsers
# -----------------------------------------------------------------------------


def parse_client_metrics_file(file_path: Path) -> Dict[str, float]:
    """Parses open-loop client_metrics CSV and converts ns to milliseconds (ms)."""
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:

            def get_ms(col_name: str) -> float:
                val = row.get(col_name)
                # Convert nanoseconds (ns) to milliseconds (ms) for open-loop response time
                return (float(val) /
                        1_000_000.0 if val is not None and val != "" else 0.0)

            return {
                "target_rate":
                float(row.get("target_rate_ops_sec", 0)),
                "achieved_throughput":
                float(row.get("achieved_throughput_ops_sec", 0)),
                "total_ops":
                float(row.get("total_ops", 0)),
                "duration_secs":
                float(row.get("duration_secs", 0)),
                "p50_ms":
                get_ms("p50_ns"),
                "p90_ms":
                get_ms("p90_ns"),
                "p95_ms":
                get_ms("p95_ns"),
                "p99_ms":
                get_ms("p99_ns"),
                "p99_9_ms":
                get_ms("p99_9_ns"),
                "p99_99_ms":
                get_ms("p99_99_ns"),
                "max_ms":
                get_ms("max_ns"),
            }
    return {}


# -----------------------------------------------------------------------------
# 3. Statistical Functions
# -----------------------------------------------------------------------------


def compute_ci95(values: List[float]) -> Tuple[float, float]:
    """Computes Mean and 95% Confidence Interval Half-Width."""
    if not values:
        return 0.0, 0.0
    m = mean(values)
    n = len(values)
    if n < 2:
        return m, 0.0

    std_err = stdev(values) / math.sqrt(n)
    return m, 1.96 * std_err


def format_stat(m: float, ci: float, precision: int = 2) -> str:
    """Formats stat as 'Mean ± CI' or 'Mean' if single sample."""
    if ci > 0.0:
        return f"{m:.{precision}f} ± {ci:.{precision}f}"
    return f"{m:.{precision}f}"


# -----------------------------------------------------------------------------
# 4. Open-Loop Result Processing Engine
# -----------------------------------------------------------------------------


def process_open_loop_directory(
    experiment_dir: Path,
) -> Tuple[Dict[int, List[Dict]], Dict[Tuple[str, str], Dict[float, Dict]]]:
    """Iterates through experiment subfolders and aggregates measurements across rate sweeps."""
    raw_data: Dict[int, Dict[Tuple[str, str],
                             Dict[float, Dict[str, List[float]]]]] = {}

    for folder in sorted(experiment_dir.iterdir()):
        if not folder.is_dir():
            continue

        strategy, threshold_str = parse_folder_name(folder.name)

        for metrics_file in folder.glob(
                "**/client_metrics_rep_*_t*_rate*.csv"):
            meta = parse_filename_metadata(metrics_file.name)
            if not meta:
                continue

            rep, threads, target_rate = meta
            client_metrics = parse_client_metrics_file(metrics_file)
            if not client_metrics:
                continue

            threads_dict = raw_data.setdefault(threads, {})
            strat_dict = threads_dict.setdefault((strategy, threshold_str), {})
            rate_group = strat_dict.setdefault(
                target_rate,
                {
                    "achieved_tp": [],
                    "p50_ms": [],
                    "p90_ms": [],
                    "p95_ms": [],
                    "p99_ms": [],
                    "p99_9_ms": [],
                    "p99_99_ms": [],
                    "max_ms": [],
                },
            )

            rate_group["achieved_tp"].append(
                client_metrics["achieved_throughput"])
            rate_group["p50_ms"].append(client_metrics["p50_ms"])
            rate_group["p90_ms"].append(client_metrics["p90_ms"])
            rate_group["p95_ms"].append(client_metrics["p95_ms"])
            rate_group["p99_ms"].append(client_metrics["p99_ms"])
            rate_group["p99_9_ms"].append(client_metrics["p99_9_ms"])
            rate_group["p99_99_ms"].append(client_metrics["p99_99_ms"])
            rate_group["max_ms"].append(client_metrics["max_ms"])

    tables_by_threads: Dict[int, List[Dict]] = {}
    series_data: Dict[Tuple[str, str], Dict[float, Dict]] = {}

    for threads in sorted(raw_data.keys()):
        rows = []
        for (strategy, threshold_str), rates_map in raw_data[threads].items():
            for target_rate in sorted(rates_map.keys()):
                metrics = rates_map[target_rate]

                tp_m, tp_ci = compute_ci95(metrics["achieved_tp"])
                p50_m, p50_ci = compute_ci95(metrics["p50_ms"])
                p90_m, p90_ci = compute_ci95(metrics["p90_ms"])
                p95_m, p95_ci = compute_ci95(metrics["p95_ms"])
                p99_m, p99_ci = compute_ci95(metrics["p99_ms"])
                p999_m, p999_ci = compute_ci95(metrics["p99_9_ms"])
                p9999_m, p9999_ci = compute_ci95(metrics["p99_99_ms"])
                max_m, max_ci = compute_ci95(metrics["max_ms"])

                series_entry = series_data.setdefault(
                    (strategy, threshold_str), {})
                series_entry[target_rate] = {
                    "achieved_tp_m": tp_m,
                    "p50_m": p50_m,
                    "p50_ci": p50_ci,
                    "p99_m": p99_m,
                    "p99_ci": p99_ci,
                    "p9999_m": p9999_m,
                    "p9999_ci": p9999_ci,
                }

                strategy_order = 0 if strategy == "unfiltered" else 1
                try:
                    thresh_num = float(threshold_str)
                except ValueError:
                    thresh_num = 0.0

                rows.append({
                    "sort_key": (strategy_order, thresh_num, target_rate),
                    "strategy":
                    strategy,
                    "threshold":
                    threshold_str,
                    "target_rate":
                    f"{target_rate:,.0f}",
                    "achieved_tp":
                    format_stat(tp_m, tp_ci, 2),
                    "p50_str":
                    format_stat(p50_m, p50_ci, 3),
                    "p90_str":
                    format_stat(p90_m, p90_ci, 3),
                    "p95_str":
                    format_stat(p95_m, p95_ci, 3),
                    "p99_str":
                    format_stat(p99_m, p99_ci, 3),
                    "p99_9_str":
                    format_stat(p999_m, p999_ci, 3),
                    "p99_99_str":
                    format_stat(p9999_m, p9999_ci, 3),
                    "max_str":
                    format_stat(max_m, max_ci, 3),
                })

        rows.sort(key=lambda x: x["sort_key"])
        tables_by_threads[threads] = rows

    return tables_by_threads, series_data


# -----------------------------------------------------------------------------
# 5. Knee-of-the-Curve 3-Subplot Plotter (Throughput & Latency)
# -----------------------------------------------------------------------------


def generate_knee_plots(
    experiment_dir: Path,
    series_data: Dict[Tuple[str, str], Dict[float, Dict]],
):
    """Generates publication-quality PDF illustrating achieved throughput plateau and latency knees."""
    pdf_path = experiment_dir / "open_loop_knees.pdf"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # Collect all rate keys across series to determine global x-limits for the y=x reference line
    all_rates_k = set()
    for rate_data in series_data.values():
        for r in rate_data.keys():
            all_rates_k.add(r / 1000.0)

    if all_rates_k:
        min_rate_k, max_rate_k = min(all_rates_k), max(all_rates_k)
        # Plot 1:1 Ideal Reference Line (y = x)
        ax1.plot(
            [min_rate_k, max_rate_k],
            [min_rate_k, max_rate_k],
            color="gray",
            linestyle=":",
            linewidth=2.0,
            label="Ideal Load (1:1)",
        )

    sorted_series_keys = sorted(series_data.keys())

    for key in sorted_series_keys:
        rate_data = series_data[key]
        sorted_rates = sorted(rate_data.keys())

        if not sorted_rates:
            continue

        strategy, threshold_str = key

        color = METHOD_COLORS.get(strategy, CB_BLUE)
        vol_key = get_volume_key(strategy, threshold_str)
        style_info = VOLUME_STYLES.get(vol_key, VOLUME_STYLES["100%"])

        ls = style_info["ls"]
        marker = style_info["marker"]

        # Convert ops/sec to thousands (kops/sec) for clean X-axis formatting
        x_rates_k = [r / 1000.0 for r in sorted_rates]

        tp_m = [rate_data[r]["achieved_tp_m"] / 1000.0 for r in sorted_rates]

        p50_m = [rate_data[r]["p50_m"] for r in sorted_rates]
        p50_ci = [rate_data[r]["p50_ci"] for r in sorted_rates]

        p99_m = [rate_data[r]["p99_m"] for r in sorted_rates]
        p99_ci = [rate_data[r]["p99_ci"] for r in sorted_rates]

        # 1. Achieved Throughput Plot (Linear Scale)
        ax1.plot(
            x_rates_k,
            tp_m,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.5,
            markersize=8,
        )

        # 2. Median Latency Plot (p50 Log Scale)
        ax2.plot(
            x_rates_k,
            p50_m,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.5,
            markersize=8,
        )
        ax2.fill_between(
            x_rates_k,
            [max(0.001, m - ci) for m, ci in zip(p50_m, p50_ci)],
            [m + ci for m, ci in zip(p50_m, p50_ci)],
            color=color,
            alpha=0.15,
        )

        # 3. Tail Latency Plot (p99 Log Scale)
        ax3.plot(
            x_rates_k,
            p99_m,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.5,
            markersize=8,
        )
        ax3.fill_between(
            x_rates_k,
            [max(0.001, m - ci) for m, ci in zip(p99_m, p99_ci)],
            [m + ci for m, ci in zip(p99_m, p99_ci)],
            color=color,
            alpha=0.15,
        )

    # Subplot Titles & Axis Formatting
    ax1.set_title(r"\textbf{Achieved Throughput}", pad=12)
    ax1.set_xlabel("Offered Load (1000 ops/sec)")
    ax1.set_ylabel("Achieved Throughput (1000 ops/sec)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_title(r"\textbf{Median Latency ($p_{50}$)}", pad=12)
    ax2.set_xlabel("Offered Load (1000 ops/sec)")
    ax2.set_ylabel("Response Latency (ms)")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    ax3.set_title(r"\textbf{Tail Latency ($p_{99}$)}", pad=12)
    ax3.set_xlabel("Offered Load (1000 ops/sec)")
    ax3.set_ylabel("Response Latency (ms)")
    ax3.set_yscale("log")
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)

    # Decoupled Legend 1: Methods (Colors)
    method_handles = [
        mlines.Line2D(
            [],
            [],
            color=METHOD_COLORS["unfiltered"],
            lw=3,
            label="Unfiltered Baseline",
        ),
        mlines.Line2D(
            [],
            [],
            color=METHOD_COLORS["persistence_path"],
            lw=3,
            label="Persistence Path Control",
        ),
    ]

    # Decoupled Legend 2: Included Volume (Line Styles in Black)
    volume_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            ls=VOLUME_STYLES["23%"]["ls"],
            marker=VOLUME_STYLES["23%"]["marker"],
            lw=2,
            markersize=8,
            label=VOLUME_STYLES["23%"]["label"],
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            ls=VOLUME_STYLES["40%"]["ls"],
            marker=VOLUME_STYLES["40%"]["marker"],
            lw=2,
            markersize=8,
            label=VOLUME_STYLES["40%"]["label"],
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            ls=VOLUME_STYLES["85%"]["ls"],
            marker=VOLUME_STYLES["85%"]["marker"],
            lw=2,
            markersize=8,
            label=VOLUME_STYLES["85%"]["label"],
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            ls=VOLUME_STYLES["100%"]["ls"],
            marker=VOLUME_STYLES["100%"]["marker"],
            lw=2,
            markersize=8,
            label=VOLUME_STYLES["100%"]["label"],
        ),
    ]

    title_methods = (r"\textbf{Strategy (Color)}"
                     if HAS_TEX else "Strategy (Color)")
    title_volume = (r"\textbf{Included Volume (Line Style)}"
                    if HAS_TEX else "Included Volume (Line Style)")

    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.26, 1.15),
        ncol=2,
        frameon=True,
        title=title_methods,
    )

    fig.legend(
        handles=volume_handles,
        loc="upper center",
        bbox_to_anchor=(0.68, 1.15),
        ncol=4,
        frameon=True,
        title=title_volume,
    )

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\n[Plotter] Open-loop knee curves PDF exported to: {pdf_path}")


# -----------------------------------------------------------------------------
# 6. Formatted Console Table Output
# -----------------------------------------------------------------------------


def print_formatted_tables(tables_by_threads: Dict[int, List[Dict]]):
    """Prints ASCII tables summarizing latency knees across offered load rates."""
    headers = [
        "Strategy",
        "Threshold",
        "Target (ops/s)",
        "Achieved (ops/s)",
        "p50 (ms)",
        "p90 (ms)",
        "p95 (ms)",
        "p99 (ms)",
        "p99.9 (ms)",
        "p99.99 (ms)",
        "Max (ms)",
    ]

    for threads, rows in tables_by_threads.items():
        print(f"\n{'='*140}")
        print(
            f" OPEN-LOOP LATENCY KNEES ({threads} THREAD{'S' if threads > 1 else ''}):"
        )
        print(f"{'='*140}")

        col_widths = [len(h) for h in headers]
        table_data = []

        for r in rows:
            row_data = [
                r["strategy"],
                r["threshold"],
                r["target_rate"],
                r["achieved_tp"],
                r["p50_str"],
                r["p90_str"],
                r["p95_str"],
                r["p99_str"],
                r["p99_9_str"],
                r["p99_99_str"],
                r["max_str"],
            ]
            table_data.append(row_data)
            for i, val in enumerate(row_data):
                col_widths[i] = max(col_widths[i], len(val))

        header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}"
                                 for i in range(len(headers)))
        separator = "-+-".join("-" * col_widths[i]
                               for i in range(len(headers)))

        print(header_line)
        print(separator)

        for row_data in table_data:
            line = " | ".join(f"{row_data[i]:<{col_widths[i]}}"
                              for i in range(len(row_data)))
            print(line)


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=
        "Process Open-Loop Benchmark results, print ASCII latency tables, and generate Knee-of-the-Curve PDF."
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help=
        "Path to open-loop experiment directory (e.g., ./open_loop_experiments/2026-10-07_12-00-00)",
    )

    args = parser.parse_args()
    exp_path = Path(args.experiment_dir)

    if not exp_path.exists() or not exp_path.is_dir():
        print(
            f"Error: Experiment directory '{exp_path}' does not exist or is not a folder."
        )
        return

    tables, series_data = process_open_loop_directory(exp_path)

    # 1. Print formatted console tables
    print_formatted_tables(tables)

    # 2. Export publication PDF figure
    generate_knee_plots(exp_path, series_data)


if __name__ == "__main__":
    main()
