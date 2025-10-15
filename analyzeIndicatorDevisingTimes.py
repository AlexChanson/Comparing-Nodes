#!/usr/bin/env python3
"""
Compute Spearman correlations between numeric columns after averaging by
(database, label). Saves a heatmap (PNG) and the correlation matrix (CSV).

Usage:
    python correlate_after_averaging.py --input results_07-10-25:07:27:49.csv \
                                        --outdir .
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(input_csv: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    df = pd.read_csv(input_csv)

    # --- Numeric columns to average (exclude obvious non-features) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {c for c in numeric_cols if c.lower() == "run" or c.lower().startswith("unnamed")}
    numeric_cols_to_avg = [c for c in numeric_cols if c not in exclude]

    # --- Average by (database, label) ---
    grouped = (
        df.groupby(["database", "label"], dropna=False)[numeric_cols_to_avg]
          .mean()
          .reset_index()
    )

    # --- Correlation (Spearman) over aggregated numeric columns ---
    corr_cols = grouped.select_dtypes(include=[np.number]).columns.tolist()
    corr = grouped[corr_cols].corr(method="spearman")

    # --- Save correlation table ---
    corr_csv = outdir / "spearman_correlation_matrix.csv"
    corr.to_csv(corr_csv, index=True)

    # --- Plot heatmap with matplotlib (no seaborn, no explicit colors) ---
    n = len(corr_cols)
    fig_w = max(8, 0.5 * n)
    fig_h = max(6, 0.5 * n)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(corr, aspect="auto")  # default colormap
    plt.colorbar(im)
    plt.xticks(ticks=np.arange(n), labels=corr_cols, rotation=90)
    plt.yticks(ticks=np.arange(n), labels=corr_cols)

    # annotate cells
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    plt.title("Spearman Correlation Heatmap (averaged by database & label)")
    plt.tight_layout()
    heatmap_png = outdir / "spearman_correlation_heatmap.png"
    plt.savefig(heatmap_png, dpi=200)
    # plt.show()  # uncomment if you want an interactive window

    print(f"[OK] Saved: {heatmap_png}")
    print(f"[OK] Saved: {corr_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Path to the input CSV")
    p.add_argument("--outdir", type=Path, default=Path("."), help="Directory to write outputs")
    args = p.parse_args()
    main(args.input, args.outdir)
