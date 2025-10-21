from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pushdown_by_label(csv_path: str) -> None:
    # Load
    df = pd.read_csv(csv_path)

    # 1) Normalize column names (strip + lowercase)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # 2) Required columns
    required = {"run", "pushdown", "ratio_prop_dropped", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    # 3) Coerce pushdown to bool
    if df["pushdown"].dtype == object:
        df["pushdown"] = (
            df["pushdown"].astype(str).str.strip().str.lower()
              .map({"true": True, "false": False, "1": True, "0": False})
        )
    else:
        df["pushdown"] = df["pushdown"].astype(int).astype(bool)

    # 4) Find and coerce time_* columns to numeric
    time_cols = [c for c in df.columns if c.startswith("time_")]
    for c in time_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Map capitalization variants to canonical lowercase names
    canon_map = {}
    for c in time_cols:
        lc = c.lower()
        if lc in {"time_total", "time_indicators", "time_validation"} and lc != c:
            canon_map[c] = lc
    if canon_map:
        df = df.rename(columns=canon_map)

    time_cols = [c for c in df.columns if c.startswith("time_")]
    for m in ("time_total", "time_indicators", "time_validation"):
        if m not in time_cols:
            raise ValueError(f"Required time column '{m}' not found. Available: {time_cols}")

    # 5) Aggregate: mean & std over runs for each (label, pushdown, ratio)
    keys = ["label", "pushdown", "ratio_prop_dropped"]
    mean_df = (df.groupby(keys, as_index=False)[time_cols]
                 .mean()
                 .rename(columns={c: f"{c}_mean" for c in time_cols}))
    std_df  = (df.groupby(keys, as_index=False)[time_cols]
                 .std(ddof=1)
                 .rename(columns={c: f"{c}_std" for c in time_cols}))
    agg = (mean_df.merge(std_df, on=keys, how="inner")
                  .sort_values(keys)
                  .reset_index(drop=True))

    # 6) Build tidy frame just for the 3 requested metrics
    records = []
    for metric in ("time_total", "time_indicators", "time_validation"):
        m_mean, m_std = f"{metric}_mean", f"{metric}_std"
        if m_mean not in agg.columns or m_std not in agg.columns:
            raise ValueError(f"Expected '{m_mean}' and '{m_std}' in aggregated data.")
        for _, row in agg.iterrows():
            records.append({
                "label": row["label"],
                "pushdown": row["pushdown"],
                "ratio": row["ratio_prop_dropped"],
                "mean": row[m_mean],
                "std": 0.0 if pd.isna(row[m_std]) else row[m_std],
                "metric": metric
            })
    tidy = pd.DataFrame.from_records(records)

    # 7) Plot helper
    def _lineplot_with_errorbars(plot_df, metric, filename, title):
        subset = plot_df[plot_df["metric"] == metric].copy()
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(
            data=subset,
            x="ratio", y="mean",
            hue="label",         # City vs Actor
            style="pushdown",    # True vs False
            markers=True, dashes=True
        )
        # Std error bars per (label, pushdown)
        for (lbl, pdwn), sub in subset.groupby(["label", "pushdown"]):
            sub = sub.sort_values("ratio")
            plt.errorbar(
                sub["ratio"], sub["mean"], yerr=sub["std"],
                fmt="none", capsize=4, linewidth=0.8, alpha=0.9
            )
        ax.set_xlabel("ratio_prop_dropped")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(title="label / pushdown", loc="best")
        plt.tight_layout()
        out = Path(csv_path).resolve().parent / filename
        plt.savefig(out, dpi=200)
        plt.close()
        return out

    # 8) Produce the three figures
    _lineplot_with_errorbars(
        tidy, "time_total",
        "time_total_by_ratio_label_pushdown.png",
        "time_total vs ratio_prop_dropped (±1 SD) by label & pushdown"
    )
    _lineplot_with_errorbars(
        tidy, "time_indicators",
        "time_indicators_by_ratio_label_pushdown.png",
        "time_indicators vs ratio_prop_dropped (±1 SD) by label & pushdown"
    )
    _lineplot_with_errorbars(
        tidy, "time_validation",
        "time_validation_by_ratio_label_pushdown.png",
        "time_validation vs ratio_prop_dropped (±1 SD) by label & pushdown"
    )

def main(file):
    # Default to the uploaded filename if present
    #default_csv = "results_test_pushdown.csv"
    plot_pushdown_by_label(file)

# Optional CLI entry point (keeps the required single-parameter function intact)
if __name__ == "__main__":
    main('reports/pushdown_results_city_airport.csv')

