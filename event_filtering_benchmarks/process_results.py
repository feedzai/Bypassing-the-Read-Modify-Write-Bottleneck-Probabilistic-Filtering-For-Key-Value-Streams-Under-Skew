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
CB_PURPLE, CB_YELLOW = "#CC79A7", "#D4AC0D"  # Crisp Gold/Yellow for Fixed Rate

# Strategy Color Map (Fixed color per method)
METHOD_COLORS = {
    "unfiltered": CB_BLUE,
    "batched": CB_PURPLE,
    "persistence_path": CB_ORANGE,
    "full_stream_filtering": CB_GREEN,
    "fixed_rate_filtering": CB_YELLOW,
}

# Volume Line Style Map (Consistent across methods)
VOLUME_STYLES = {
    "23%": {
        "ls": ":",
        "marker": "^",
        "label": r"$\sim 23\%$ Vol"
    },
    "40%": {
        "ls": "-.",
        "marker": "s",
        "label": r"$\sim 40\%$ Vol"
    },
    "85%": {
        "ls": "--",
        "marker": "v",
        "label": r"$\sim 85\%$ Vol"
    },
    "100%": {
        "ls": "-",
        "marker": "o",
        "label": r"$100\%$ Vol"
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
    if folder_name in ("unfiltered", "batched"):
        return folder_name, "-"

    prefixes = [
        "persistence_path",
        "full_stream_filtering",
        "fixed_rate_filtering",
    ]
    for prefix in prefixes:
        if folder_name.startswith(f"{prefix}_"):
            threshold_str = folder_name[len(prefix) + 1:]
            return prefix, threshold_str

    parts = folder_name.rsplit("_", 1)
    if len(parts) == 2 and re.match(r"^\d+(\.\d+)?$", parts[1]):
        return parts[0], parts[1]

    return folder_name, "-"


def parse_filename_metadata(filename: str) -> Optional[Tuple[int, int]]:
    """Extracts repetition ID and thread count from CSV filename."""
    pattern = r".*_rep_(\d+)_t(\d+)\.csv$"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def get_volume_key(strategy: str, threshold_str: str) -> str:
    """Maps (strategy, threshold) pair to standard volume percentage bucket."""
    if strategy in ("unfiltered", "batched"):
        return "100%"

    if strategy == "fixed_rate_filtering":
        if threshold_str in ("0.23", "0.25", "0.2"):
            return "23%"
        elif threshold_str in ("0.40", "0.4", "0.5"):
            return "40%"
        elif threshold_str in ("0.85", "0.8"):
            return "85%"
        elif threshold_str in ("1.0", "1.00"):
            return "100%"
    else:  # velocity methods (persistence_path, full_stream_filtering)
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
    """Parses client_metrics CSV and converts nanoseconds (ns) to microseconds (µs)."""
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:

            def get_us(col_name: str) -> float:
                val = row.get(col_name)
                return (float(val) /
                        1000.0 if val is not None and val != "" else 0.0)

            return {
                "total_ops": float(row.get("total_ops", 0)),
                "duration_secs": float(row.get("duration_secs", 0)),
                "throughput": float(row.get("throughput_ops_sec", 0)),
                "p50_us": get_us("p50_ns"),
                "p90_us": get_us("p90_ns"),
                "p95_us": get_us("p95_ns"),
                "p99_us": get_us("p99_ns"),
                "p99_9_us": get_us("p99_9_ns"),
                "p99_99_us": get_us("p99_99_ns"),
                "max_us": get_us("max_ns"),
            }
    return {}


def parse_db_stats_file(file_path: Path) -> Dict[str, float]:
    """Parses db_stats CSV to extract write ops, storage sizes, and WAF metrics."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:

            def get_val(keys: List[str]) -> float:
                for k in keys:
                    if k in row and row[k] != "":
                        return float(row[k])
                return 0.0

            total_write_ops = get_val(
                ["total_write_ops", "total_writes", "mutations"])

            flush_gb = get_val(["flush_gb"])
            if flush_gb == 0.0 and "bytes_flushed" in row:
                flush_gb = get_val(["bytes_flushed"]) / (1024**3)

            disk_gb = get_val(["total_disk_gb"])
            if disk_gb == 0.0 and "total_disk_bytes" in row:
                disk_gb = get_val(["total_disk_bytes"]) / (1024**3)

            return {
                "total_write_ops": total_write_ops,
                "flush_gb": flush_gb,
                "total_disk_gb": disk_gb,
                "compaction_waf": get_val(["compaction_waf", "comp_waf"]),
                "system_waf": get_val(["system_waf", "sys_waf"]),
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
# 4. Result Processing Engine
# -----------------------------------------------------------------------------


def process_experiment_directory(
    experiment_dir: Path,
) -> Tuple[Dict[int, List[Dict]], Dict[Tuple[str, str], Dict[int, Dict]]]:
    """Iterates through experiment subfolders and aggregates measurements across repetitions."""
    raw_data: Dict[int, Dict[Tuple[str, str], Dict[str, List[float]]]] = {}

    for folder in sorted(experiment_dir.iterdir()):
        if not folder.is_dir():
            continue

        strategy, threshold_str = parse_folder_name(folder.name)

        for metrics_file in folder.glob("client_metrics_rep_*_t*.csv"):
            meta = parse_filename_metadata(metrics_file.name)
            if not meta:
                continue

            rep, threads = meta
            client_metrics = parse_client_metrics_file(metrics_file)

            db_stats_file = folder / f"db_stats_rep_{rep}_t{threads}.csv"
            db_metrics = parse_db_stats_file(db_stats_file)

            total_ops = client_metrics.get("total_ops", 0)
            total_write_ops = db_metrics.get("total_write_ops", 0)

            if strategy in ("unfiltered", "batched"):
                included_vol_pct = 100.0
            elif total_write_ops > 0 and total_ops > 0:
                included_vol_pct = (total_write_ops / total_ops) * 100.0
            elif strategy == "fixed_rate_filtering":
                try:
                    included_vol_pct = float(threshold_str) * 100.0
                except ValueError:
                    included_vol_pct = 0.0
            else:
                included_vol_pct = 0.0

            threads_dict = raw_data.setdefault(threads, {})
            group = threads_dict.setdefault(
                (strategy, threshold_str),
                {
                    "vol_pct": [],
                    "throughput": [],
                    "p50_us": [],
                    "p90_us": [],
                    "p95_us": [],
                    "p99_us": [],
                    "p99_9_us": [],
                    "p99_99_us": [],
                    "max_us": [],
                    "flush_gb": [],
                    "total_disk_gb": [],
                    "compaction_waf": [],
                    "system_waf": [],
                },
            )

            group["vol_pct"].append(included_vol_pct)
            group["throughput"].append(client_metrics.get("throughput", 0.0))
            group["p50_us"].append(client_metrics.get("p50_us", 0.0))
            group["p90_us"].append(client_metrics.get("p90_us", 0.0))
            group["p95_us"].append(client_metrics.get("p95_us", 0.0))
            group["p99_us"].append(client_metrics.get("p99_us", 0.0))
            group["p99_9_us"].append(client_metrics.get("p99_9_us", 0.0))
            group["p99_99_us"].append(client_metrics.get("p99_99_us", 0.0))
            group["max_us"].append(client_metrics.get("max_us", 0.0))

            group["flush_gb"].append(db_metrics.get("flush_gb", 0.0))
            group["total_disk_gb"].append(db_metrics.get("total_disk_gb", 0.0))
            group["compaction_waf"].append(
                db_metrics.get("compaction_waf", 0.0))
            group["system_waf"].append(db_metrics.get("system_waf", 0.0))

    tables_by_threads: Dict[int, List[Dict]] = {}
    series_data: Dict[Tuple[str, str], Dict[int, Dict]] = {}

    for threads in sorted(raw_data.keys()):
        rows = []
        for (strategy, threshold_str), metrics in raw_data[threads].items():
            vol_m, vol_ci = compute_ci95(metrics["vol_pct"])
            tp_m, tp_ci = compute_ci95(metrics["throughput"])
            p50_m, p50_ci = compute_ci95(metrics["p50_us"])
            p90_m, p90_ci = compute_ci95(metrics["p90_us"])
            p95_m, p95_ci = compute_ci95(metrics["p95_us"])
            p99_m, p99_ci = compute_ci95(metrics["p99_us"])
            p999_m, p999_ci = compute_ci95(metrics["p99_9_us"])
            p9999_m, p9999_ci = compute_ci95(metrics["p99_99_us"])
            max_m, max_ci = compute_ci95(metrics["max_us"])

            flush_m, flush_ci = compute_ci95(metrics["flush_gb"])
            disk_m, disk_ci = compute_ci95(metrics["total_disk_gb"])
            cwaf_m, cwaf_ci = compute_ci95(metrics["compaction_waf"])
            swaf_m, swaf_ci = compute_ci95(metrics["system_waf"])

            series_entry = series_data.setdefault((strategy, threshold_str),
                                                  {})
            series_entry[threads] = {
                "tp_m": tp_m,
                "tp_ci": tp_ci,
                "p50_m": p50_m,
                "p50_ci": p50_ci,
                "p9999_m": p9999_m,
                "p9999_ci": p9999_ci,
            }

            strategy_order = {
                "unfiltered": 0,
                "batched": 1,
                "persistence_path": 2,
                "full_stream_filtering": 3,
                "fixed_rate_filtering": 4,
            }.get(strategy, 99)

            try:
                thresh_num = float(threshold_str)
            except ValueError:
                thresh_num = 0.0

            rows.append({
                "sort_key": (strategy_order, thresh_num),
                "strategy": strategy,
                "threshold": threshold_str,
                "vol_pct_str": f"{vol_m:.1f}%",
                "throughput_str": format_stat(tp_m, tp_ci, 2),
                "p50_str": format_stat(p50_m, p50_ci, 2),
                "p90_str": format_stat(p90_m, p90_ci, 2),
                "p95_str": format_stat(p95_m, p95_ci, 2),
                "p99_str": format_stat(p99_m, p99_ci, 2),
                "p99_9_str": format_stat(p999_m, p999_ci, 2),
                "p99_99_str": format_stat(p9999_m, p9999_ci, 2),
                "max_str": format_stat(max_m, max_ci, 2),
                "flush_gb_str": format_stat(flush_m, flush_ci, 3),
                "disk_gb_str": format_stat(disk_m, disk_ci, 3),
                "comp_waf_str": format_stat(cwaf_m, cwaf_ci, 2),
                "sys_waf_str": format_stat(swaf_m, swaf_ci, 2),
            })

        rows.sort(key=lambda x: x["sort_key"])
        tables_by_threads[threads] = rows

    return tables_by_threads, series_data


# -----------------------------------------------------------------------------
# 5. Page-Wide 3-Subplot Generation Function (Decoupled Color & Style)
# -----------------------------------------------------------------------------


def generate_plots(
    experiment_dir: Path,
    series_data: Dict[Tuple[str, str], Dict[int, Dict]],
):
    """Generates page-wide 3-subplot PDF figure with decoupled strategy color and volume line styles."""
    pdf_path = experiment_dir / "benchmark_metrics.pdf"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), sharex=True)

    sorted_series_keys = sorted(series_data.keys())

    for key in sorted_series_keys:
        thread_data = series_data[key]
        sorted_threads = sorted(thread_data.keys())

        if not sorted_threads:
            continue

        strategy, threshold_str = key

        color = METHOD_COLORS.get(strategy, CB_BLUE)
        vol_key = get_volume_key(strategy, threshold_str)
        style_info = VOLUME_STYLES.get(vol_key, VOLUME_STYLES["100%"])

        ls = style_info["ls"]
        marker = style_info["marker"]

        tp_m = [thread_data[t]["tp_m"] for t in sorted_threads]
        tp_ci = [thread_data[t]["tp_ci"] for t in sorted_threads]

        p50_m = [thread_data[t]["p50_m"] for t in sorted_threads]
        p50_ci = [thread_data[t]["p50_ci"] for t in sorted_threads]

        p9999_m = [thread_data[t]["p9999_m"] for t in sorted_threads]
        p9999_ci = [thread_data[t]["p9999_ci"] for t in sorted_threads]

        # 1. Throughput Plot
        ax1.plot(
            sorted_threads,
            tp_m,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.5,
            markersize=8,
        )
        ax1.fill_between(
            sorted_threads,
            [m - ci for m, ci in zip(tp_m, tp_ci)],
            [m + ci for m, ci in zip(tp_m, tp_ci)],
            color=color,
            alpha=0.15,
        )

        # 2. Median Latency Plot (p50)
        ax2.plot(
            sorted_threads,
            p50_m,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.5,
            markersize=8,
        )
        ax2.fill_between(
            sorted_threads,
            [m - ci for m, ci in zip(p50_m, p50_ci)],
            [m + ci for m, ci in zip(p50_m, p50_ci)],
            color=color,
            alpha=0.15,
        )

        # 3. Tail Latency Plot (p99.99)
        ax3.plot(
            sorted_threads,
            p9999_m,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.5,
            markersize=8,
        )
        ax3.fill_between(
            sorted_threads,
            [m - ci for m, ci in zip(p9999_m, p9999_ci)],
            [m + ci for m, ci in zip(p9999_m, p9999_ci)],
            color=color,
            alpha=0.15,
        )

    # Subplot Titles & Axis Formatting
    ax1.set_title(r"\textbf{Throughput}", pad=12)
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Throughput (ops/sec)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_title(r"\textbf{Median Latency ($p_{50}$)}", pad=12)
    ax2.set_xlabel("Number of Threads")
    ax2.set_ylabel(r"Latency ($\mu$s)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.set_title(r"\textbf{Tail Latency ($p_{99.99}$)}", pad=12)
    ax3.set_xlabel("Number of Threads")
    ax3.set_ylabel(r"Latency ($\mu$s)")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Decoupled Legend 1: Methods (Colors)
    method_handles = [
        mlines.Line2D(
            [],
            [],
            color=METHOD_COLORS["unfiltered"],
            lw=3,
            label="Unfiltered",
        ),
        mlines.Line2D([], [],
                      color=METHOD_COLORS["batched"],
                      lw=3,
                      label="Batched"),
        mlines.Line2D(
            [],
            [],
            color=METHOD_COLORS["persistence_path"],
            lw=3,
            label="Persistence Path",
        ),
        mlines.Line2D(
            [],
            [],
            color=METHOD_COLORS["full_stream_filtering"],
            lw=3,
            label="Full Stream Filter",
        ),
        mlines.Line2D(
            [],
            [],
            color=METHOD_COLORS["fixed_rate_filtering"],
            lw=3,
            label="Fixed Rate Filter",
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
        bbox_to_anchor=(0.33, 1.15),
        ncol=5,
        frameon=True,
        title=title_methods,
    )

    fig.legend(
        handles=volume_handles,
        loc="upper center",
        bbox_to_anchor=(0.80, 1.15),
        ncol=4,
        frameon=True,
        title=title_volume,
    )

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\n[Plotter] Page-wide PDF figure exported to: {pdf_path}")


# -----------------------------------------------------------------------------
# 6. Formatted ASCII Table Output
# -----------------------------------------------------------------------------


def print_formatted_tables(tables_by_threads: Dict[int, List[Dict]]):
    """Prints ASCII tables grouped by thread count."""
    headers = [
        "Strategy",
        "Threshold",
        "Included Vol",
        "Throughput (ops/s)",
        "p50 (µs)",
        "p90 (µs)",
        "p95 (µs)",
        "p99 (µs)",
        "p99.9 (µs)",
        "p99.99 (µs)",
        "Max (µs)",
        "Flush (GB)",
        "Total Disk (GB)",
        "Comp WAF",
        "Sys WAF",
    ]

    for threads, rows in tables_by_threads.items():
        print(f"\n{'='*165}")
        print(f" {threads} THREAD{'S' if threads > 1 else ''}:")
        print(f"{'='*165}")

        col_widths = [len(h) for h in headers]
        table_data = []

        for r in rows:
            row_data = [
                r["strategy"],
                r["threshold"],
                r["vol_pct_str"],
                r["throughput_str"],
                r["p50_str"],
                r["p90_str"],
                r["p95_str"],
                r["p99_str"],
                r["p99_9_str"],
                r["p99_99_str"],
                r["max_str"],
                r["flush_gb_str"],
                r["disk_gb_str"],
                r["comp_waf_str"],
                r["sys_waf_str"],
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
        "Process Event Filtering Benchmark results, generate ASCII tables and decoupled 3-subplot PDF."
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help=
        "Path to experiment directory (e.g., ./experiment_results/2026-10-02_17-41-26)",
    )

    args = parser.parse_args()
    exp_path = Path(args.experiment_dir)

    if not exp_path.exists() or not exp_path.is_dir():
        print(f"Error: Experiment directory '{exp_path}' does not exist.")
        return

    tables, series_data = process_experiment_directory(exp_path)

    # 1. Print formatted console tables
    print_formatted_tables(tables)

    # 2. Export decoupled publication PDF figure
    generate_plots(exp_path, series_data)


if __name__ == "__main__":
    main()
