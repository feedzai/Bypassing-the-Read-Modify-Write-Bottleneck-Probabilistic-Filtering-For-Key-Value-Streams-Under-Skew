"""This script processes the IIOT dataset.  The dataset parsed by the
author did not have timestamps therefore we write our own custom
processing script for the raw data

"""
import pandas as pd

import subprocess
from pathlib import Path

OUTPUT_DIR = "Define the directory for the processed data."
DATASET_DIR = "REPLACE BY YOUR DATASET DIR"


def process_pcap_directory(directory: Path):
    """Scans a directory for PCAP files and extracts features using TShark."""

    if not directory.exists():
        print(f"⚠Directory not found -> {directory}")
        return []

    print(f"\nScanning directory: {directory.name}")
    pcap_files = list(directory.rglob("*.pcap"))

    if not pcap_files:
        print(f"  No PCAP files found in {directory.name}")
        return []

    processed_files = []

    for pcap_file in pcap_files:
        output_filename = f"Fixed_{pcap_file.stem}.csv"
        output_filepath = pcap_file.parent / output_filename

        print(f"  Processing: {pcap_file.name} -> {output_filename}")

        command = [
            "tshark",
            "-r",
            str(pcap_file),
            "-T",
            "fields",
            "-E",
            "header=y",
            "-E",
            "separator=,",
            "-E",
            "quote=d",
            "-E",
            "occurrence=f",
            "-e",
            "frame.time_epoch",  # The absolute Unix timestamp
            "-e",
            "ip.src",
            "-e",
            "ip.dst",
            "-e",
            "tcp.srcport",
            "-e",
            "tcp.dstport",
            "-e",
            "udp.srcport",
            "-e",
            "udp.dstport",
            "-e",
            "tcp.flags",
            "-e",
            "tcp.len",
            "-e",
            "http.request.method",
            "-e",
            "mqtt.msgtype"
        ]

        try:
            with open(output_filepath, "w") as outfile:
                subprocess.run(command,
                               stdout=outfile,
                               stderr=subprocess.PIPE,
                               check=True)
            print("    Success")
            processed_files.append(output_filepath)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode().strip()
            print(f"    Failed! TShark Error: {error_msg}")

    return processed_files


if __name__ == '__mani__':
    base_path = Path(DATASET_DIR).expanduser().resolve()

    target_dirs = [base_path / "Attack traffic", base_path / "Normal traffic"]

    # Verify tshark is installed and accessible
    try:
        subprocess.run(["tshark", "-v"],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=True)
        print("TShark is installed and ready.")
    except FileNotFoundError:
        print(
            "Error: 'tshark' command not found. Please install Wireshark/TShark and add it to your PATH."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Start processing
    all_processed_csvs = []

    print("Starting PCAP extraction process...")

    for target_dir in target_dirs:
        # Process each directory and add the resulting CSV paths to our list
        new_csvs = process_pcap_directory(target_dir)
        all_processed_csvs.extend(new_csvs)

    print(
        f"Extraction complete! Generated {len(all_processed_csvs)} fixed CSV files."
    )
    # 1. Define your single output folder for all parquet files
    OUTPUT_DIR = Path(OUTPUT_DIR).expanduser()

    # 2. Create the folder if it doesn't exist yet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_parquet_files = []

    print(f"🚀 Processing CSVs and saving Parquets to: {OUTPUT_DIR}")

    for csv_path in all_processed_csvs:
        csv_path = Path(csv_path)
        print(f"  ⏳ Processing: {csv_path.name}")

        # Read the CSV
        df = pd.read_csv(csv_path, low_memory=False)

        # Add labels based on the directory it came from
        if "Attack traffic" in str(csv_path):
            df['Attack_label'] = 1
            # Extract attack type from filename
            attack_type = csv_path.stem.replace('Fixed_', '').replace(
                '_attack', '').replace(' ', '_')
            df['Attack_type'] = attack_type
        else:
            df['Attack_label'] = 0
            df['Attack_type'] = 'Normal'

        # Convert the epoch timestamp to a proper pandas datetime object NOW
        if 'frame.time_epoch' in df.columns:
            df['datetime'] = pd.to_datetime(df['frame.time_epoch'], unit='s')

        # 3. Define the new path: Output Directory + Original Filename + .parquet
        parquet_filename = f"{csv_path.stem}.parquet"
        parquet_path = OUTPUT_DIR / parquet_filename

        # Save to Parquet
        df.to_parquet(parquet_path, index=False)

        all_parquet_files.append(parquet_path)

        # Explicitly delete the dataframe and run garbage collection to free memory
        del df
        gc.collect()

    print(
        f"Success! Saved {len(all_parquet_files)} Parquet files to {OUTPUT_DIR}."
    )
