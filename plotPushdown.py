#!/usr/bin/env python3
"""
Plot total time by ratio_prop_dropped with pushdown split and error bars.

- Averages ALL columns whose name starts with 'time_' over runs (column 'run').
- Plots mean of time_total vs ratio_prop_dropped for pushdown False/True.
- Adds error bars (standard deviation across runs) for each point.
"""

import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected columns exist
    required = {"run", "pushdown", "ratio_prop_dropped"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Coerce types if needed
    if df["pushdown"].dtype == object:
        df["pushdown"] = df["pushdown"].astype(str).str.lower().map({"true": True, "false": False})
    return df

def aggregate_times(df: pd.DataFrame):
    # Identify all time_* columns
    time_cols = [c for c in df.columns if c.startswith("time_")]
    if "time_total" not in time_cols:
        raise ValueError("Column 'time_total' not found among time_* columns.")
    group_keys = ["ratio_prop_dropped", "pushdown"]

    mean_df = (
        df.groupby(group_keys, as_index=False)[time_cols]
          .mean()
          .rename(columns={c: f"{c}_mean" for c in time_cols})
    )
    std_df = (
        df.groupby(group_keys, as_index=False)[time_cols]
          .std(ddof=1)
          .rename(columns={c: f"{c}_std" for c in time_cols})
    )
    agg = pd.merge(mean_df, std_df, on=group_keys, how="inner")
    # Sort by x then by pushdown for a clean plot order
    agg = agg.sort_values(["ratio_prop_dropped", "pushdown"]).reset_index(drop=True)
    return agg, time_cols

def plot_time_total(agg: pd.DataFrame, outpath: Path, title: str = None):
    # Prepare plotting frame for seaborn (mean as y)
    plot_df = agg.rename(
        columns={
            "time_total_mean": "time_total",
            "ratio_prop_dropped": "ratio"
        }
    ).copy()
    # Seaborn lineplot for means
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=plot_df,
        x="ratio",
        y="time_total",
        hue="pushdown",
        marker="o"
    )

    # Add error bars from std (matplotlib), matching each hue series
    for push_val, sub in plot_df.groupby("pushdown"):
        sub = sub.sort_values("ratio")
        # yerr pulls from the std column in the original aggregated df
        yerr = agg.loc[sub.index, "time_total_std"].to_numpy()
        plt.errorbar(
            sub["ratio"].to_numpy(),
            sub["time_total"].to_numpy(),
            yerr=yerr,
            fmt="none",
            capsize=4,
            linewidth=0.8,
            alpha=0.9
        )

    ax.set_xlabel("ratio_prop_dropped (%)" if plot_df["ratio"].max() <= 100 else "ratio_prop_dropped")
    ax.set_ylabel("Total time (time_total)")
    if title:
        ax.set_title(title)
    ax.legend(title="pushdown", loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():

    df = load_data('reports/results_test_pushdown_1.csv')
    agg, time_cols = aggregate_times(df)

    # (Optional) print a quick summary of the aggregated time_* means
    summary_cols = ["ratio_prop_dropped", "pushdown"] + [c for c in agg.columns if c.endswith("_mean")]
    print("\nAveraged time_* columns over runs (means):")
    print(agg[summary_cols].to_string(index=False))

    plot_time_total(
        agg,
        Path('reports/results_test_pushdown.png'),
        title="time_total vs ratio_prop_dropped (±1 SD across runs)"
    )
    print(f"\nSaved plot to: reports/results_test_pushdown_1.pdf")

if __name__ == "__main__":
    main()
