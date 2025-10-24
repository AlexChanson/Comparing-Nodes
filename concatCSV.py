#!/usr/bin/env python3
"""
Concatenate all CSV files in a directory into one CSV file.

Usage:
    python concat_csvs_in_dir.py --dir path/to/csvs --out combined.csv
"""

import glob
from pathlib import Path

import pandas as pd


def concat_csvs_in_dir(directory: str, outpath: str = "combined.csv") -> None:
    """
    Concatenate all CSV files in a directory into one CSV file.

    Parameters
    ----------
    directory : str
        Path to the directory containing CSV files.
    outpath : str
        Output CSV file path (default: combined.csv in current directory).

    """
    directory = Path(directory)
    if not directory.exists():
        msg = f"Directory not found: {directory}"
        raise FileNotFoundError(msg)

    csv_files = sorted(glob.glob(str(directory / "*.csv")))
    if not csv_files:
        msg = f"No CSV files found in: {directory}"
        raise FileNotFoundError(msg)

    print(f"Found {len(csv_files)} CSV files. Reading and concatenating...")

    # Read all CSVs into list of DataFrames
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = Path(file).name  # Keep track of origin
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")

    if not dfs:
        msg = "No valid CSV files could be read."
        raise ValueError(msg)

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    combined.to_csv(outpath, index=False)
    print(f"\n✅ Combined CSV written to: {outpath}")
    print(f"Total rows: {len(combined)}, columns: {len(combined.columns)}")


if __name__ == "__main__":
    concat_csvs_in_dir('reports/pushdown results city airport', 'reports/pushdown_results_city_airport.csv')
