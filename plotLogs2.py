#!/usr/bin/env python3

"""
Parse experiment logs named: data_heuristic_k_run.log
Heuristics: lp, ls, rd, sls, exp
Extract: |D|, n, CPU time, Wall time, best score (after '|'), best clustering.
Aggregate: mean/std over runs.

Plots (Seaborn):
  CPU/Wall/Score — separate figures for:
    1) by heuristic and data  (avg across k/|D|/n inside each (heuristic,data), std as error bars)
    2) by heuristic overall   (avg across all data/k; std across those groups) — heuristics ordered by increasing time
    3) by k and heuristic     (avg across data; std)
    4) by |D| and heuristic   (avg across data; std)
    5) by n and heuristic     (avg across data; std)

Score range per data: normalized [0,1] + markers for heuristics (with and without error bars),
and a clean scatter overview.

ARI: mean pairwise ARI across runs for (data, heuristic, k); then by (data, heuristic) and by heuristic overall.

This version:
- Error bars on log-scale charts use multiplicative (log-space) extents and a distinct color.
- Avoids duplicate legends on catplots.
- Orders CPU/Wall “by heuristic” charts by increasing mean time.
- Adds box plots by (heuristic, k) for each dataset.
"""

import argparse
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =========================
# GLOBAL PARAMETERS (edit)
# =========================
PLOT_DPI = 220
STYLE = "whitegrid"
PALETTE = "deep"

# Auto log-scale: if all bar means > 0 and max/min >= LOG_THRESHOLD -> log y
LOG_THRESHOLD = 50
EPS_POS = 1e-12

# Error bar styling
ERROR_COLOR = "black"  # distinct from bar colors for readability
ERROR_LINEWIDTH = 1.2
ERROR_CAPSIZE = 4
# Relative error cap when converting to multiplicative log-space error
REL_ERR_CAP = 2.0  # 200%

HEURISTIC_ORDER = ["lp", "ls", "rd", "sls", "exp"]
HEURISTIC_LABELS = {
    "lp": "Laplacian",
    "ls": "Local search (random init)",
    "rd": "Random",
    "sls": "Greedy local search",
    "exp": "Exponential",
}

XTEXT = {
    "heuristic": "Heuristic",
    "data": "Data",
    "k": "Number of clusters (k)",
    "D": "Number of indicators (|D|)",
    "n": "Data size (n)",
}
YTEXT = {
    "cpu": "CPU time (s)",
    "wall": "Wall time (s)",
    "score": "Best solution score",
    "ari": "Mean ARI (stability across runs)",
}
TITLES = {
    "cpu_by_heur_data": "CPU time by heuristic and data",
    "cpu_by_heur_overall": "CPU time by heuristic (averaged over all data/k)",
    "cpu_by_k_heur": "CPU time by k and heuristic",
    "cpu_by_D_heur": "CPU time by |D| and heuristic",
    "cpu_by_n_heur": "CPU time by n and heuristic",
    "wall_by_heur_data": "Wall time by heuristic and data",
    "wall_by_heur_overall": "Wall time by heuristic (averaged over all data/k)",
    "wall_by_k_heur": "Wall time by k and heuristic",
    "wall_by_D_heur": "Wall time by |D| and heuristic",
    "wall_by_n_heur": "Wall time by n and heuristic",
    "score_by_heur_data": "Best score by heuristic and data",
    "score_by_heur_overall": "Best score by heuristic (averaged over all data/k)",
    "score_by_k_heur": "Best score by k and heuristic",
    "score_by_D_heur": "Best score by |D| and heuristic",
    "score_by_n_heur": "Best score by n and heuristic",
    "range_plot": "Score range per data with heuristic positions",
    "ari_by_data_heur": "Mean ARI by data and heuristic",
    "ari_by_heur": "Mean ARI by heuristic (± std across data/k)",
    "score_box_k_heur": "Best score distribution by k and heuristic (box plots)",
    "score_box_k_heur_per_data": "Best score distribution by k and heuristic — {}",
}

FILENAME_REGEX = re.compile(r"^(?P<data>[a-zA-Z0-9_-]+)_(?P<heuristic>lp|ls|rd|sls|exp)_(?P<k>\d+)_(?P<run>\d+)\.log$")


# =========================
# Helpers: parsing & ARI
# =========================
def parse_clusters(text: str):
    m = re.search(r"\[best solution\]\s*:\s*\[([^\]]+)\]", text)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(";")]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            return None
    return out


def parse_best_score(text: str):
    m = re.search(r"\[best solution\][^\|]*\|\s*(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def parse_cpu(text: str):
    m = re.search(r"\[CPU time\]\s*([0-9.]+)\s*seconds", text)
    return float(m.group(1)) if m else None


def parse_wall(text: str):
    m = re.search(r"\[Wall time\]\s*([0-9.]+)\s*seconds", text)
    return float(m.group(1)) if m else None


def parse_loaded_D_n(text: str):
    mD = re.search(r"\|D\|\s*=\s*(\d+)", text)
    mn = re.search(r"\bn\s*=\s*(\d+)", text)
    D = int(mD.group(1)) if mD else None
    n = int(mn.group(1)) if mn else None
    return D, n


def collect_all_scores_after_bar(text: str):
    return [float(x) for x in re.findall(r"\|\s*(-?\d+(?:\.\d+)?)", text)]


def adjusted_rand_index(labels_true, labels_pred):
    if len(labels_true) != len(labels_pred):
        return np.nan
    n = len(labels_true)
    if n <= 1:
        return 1.0

    ct = defaultdict(lambda: defaultdict(int))
    rows = defaultdict(int)
    cols = defaultdict(int)
    for t, p in zip(labels_true, labels_pred):
        ct[t][p] += 1
        rows[t] += 1
        cols[p] += 1

    def comb2(x):
        return x * (x - 1) / 2

    sum_comb_c = sum(comb2(ct[t][p]) for t in ct for p in ct[t])
    sum_comb_rows = sum(comb2(v) for v in rows.values())
    sum_comb_cols = sum(comb2(v) for v in cols.values())
    total_comb = comb2(n)

    expected_index = (sum_comb_rows * sum_comb_cols) / total_comb if total_comb else 0.0
    max_index = (sum_comb_rows + sum_comb_cols) / 2.0
    denom = max_index - expected_index
    if denom == 0.0:
        return 1.0
    return (sum_comb_c - expected_index) / denom


def mean_pairwise_ari(clusterings):
    if len(clusterings) < 2:
        return np.nan
    vals = []
    for a, b in combinations(clusterings, 2):
        if a is None or b is None:
            continue
        vals.append(adjusted_rand_index(a, b))
    return float(np.mean(vals)) if vals else np.nan


# =========================
# Parsing all logs
# =========================
def parse_log_file(path: Path):
    m = FILENAME_REGEX.match(path.name)
    if not m:
        return None
    data = m.group("data")
    heuristic = m.group("heuristic")
    k = int(m.group("k"))
    run = int(m.group("run"))

    text = path.read_text(encoding="utf-8", errors="ignore")

    D, n = parse_loaded_D_n(text)
    cpu = parse_cpu(text)
    wall = parse_wall(text)
    best_score = parse_best_score(text)
    clusters = parse_clusters(text)
    all_bar_scores = collect_all_scores_after_bar(text)

    return {
        "file": str(path),
        "data": data,
        "heuristic": heuristic,
        "k": k,
        "run": run,
        "D": D,
        "n": n,
        "cpu_time": cpu,
        "wall_time": wall,
        "best_score": best_score,
        "clusters": clusters,
        "all_bar_scores": all_bar_scores,
    }


def gather_records(indir: Path):
    rows = []
    for p in sorted(indir.glob("*.log")):
        rec = parse_log_file(p)
        if rec:
            rows.append(rec)
    return pd.DataFrame(rows)


# =========================
# Aggregations & ARI
# =========================
def aggregate_runs(df: pd.DataFrame):
    group_cols = ["data", "heuristic", "k", "D", "n"]
    return (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            cpu_time_mean=("cpu_time", "mean"),
            cpu_time_std=("cpu_time", "std"),
            wall_time_mean=("wall_time", "mean"),
            wall_time_std=("wall_time", "std"),
            best_score_mean=("best_score", "mean"),
            best_score_std=("best_score", "std"),
            runs=("run", "nunique"),
        )
        .reset_index()
    )


def compute_score_range_by_data(df: pd.DataFrame):
    exp = []
    for _, row in df.iterrows():
        if isinstance(row.get("all_bar_scores"), list):
            for v in row["all_bar_scores"]:
                exp.append((row["data"], v))
    ranges = pd.DataFrame(exp, columns=["data", "score"])
    ranges = ranges.groupby("data", as_index=False)["score"].agg(["min", "max"]).reset_index()

    heur = df.groupby(["data", "heuristic"], observed=True, as_index=False).agg(
        mean_best=("best_score", "mean"), std_best=("best_score", "std")
    )
    return ranges, heur


def compute_mean_ari(df: pd.DataFrame):
    groups = df.groupby(["data", "heuristic", "k"], observed=True)
    rows = []
    for (data, h, k), g in groups:
        cls = [c for c in g["clusters"].tolist() if isinstance(c, list)]
        rows.append({"data": data, "heuristic": h, "k": k, "mean_ari": mean_pairwise_ari(cls)})
    per_k = pd.DataFrame(rows)

    by_data_heur = per_k.groupby(["data", "heuristic"], observed=True, as_index=False).agg(
        mean_ari=("mean_ari", "mean")
    )
    by_heur = per_k.groupby("heuristic", observed=True, as_index=False).agg(
        mean_ari=("mean_ari", "mean"), std_ari=("mean_ari", "std")
    )
    return per_k, by_data_heur, by_heur


# =========================
# Plotting helpers (error bars & log-scale aware)
# =========================
def _auto_log_decision(y_vals, mode="auto") -> str:
    if mode == "linear":
        return "linear"
    if mode == "log":
        return "log"
    y = np.array([v for v in y_vals if pd.notna(v)], dtype=float)
    if y.size == 0:
        return "linear"
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    if ymin <= 0:
        return "linear"
    return "log" if (ymax / max(ymin, EPS_POS)) >= LOG_THRESHOLD else "linear"


def _asym_err_for_scale(mean, std, scale_mode):
    """
    Compute lower/upper error for a given scale.
    - linear: symmetric std (lower=min(std,mean), upper=std)
    - log: multiplicative log-space extents using CV (log-normal approx), asymmetric.
    """
    if pd.isna(mean) or pd.isna(std) or std <= 0:
        return (np.nan, np.nan)

    if scale_mode == "log" and mean > 0:
        cv = max(min(std / max(mean, EPS_POS), REL_ERR_CAP), 0.0)
        tau = np.sqrt(np.log(1.0 + cv**2))
        f = np.exp(tau)
        lower = mean - (mean / f)
        upper = (mean * f) - mean
        lower = max(lower, 0.0)
        upper = max(upper, 0.0)
        return (lower, upper)
    lower = min(std, mean) if mean > 0 else std
    upper = std
    return (lower, upper)


def bar_with_std(
    df,
    *,
    x,
    mean_col,
    std_col,
    hue=None,
    order=None,
    hue_order=None,
    title="",
    xlabel="",
    ylabel="",
    out=None,
    yscale="auto",
) -> None:
    """
    Plot pre-aggregated means with std as error bars.
    Robust to missing (x,hue) combos and dropped categories (NaN means).
    Auto-log y-scale when relevant; error bars adapt to scale and use a distinct color.
    """
    sns.set_theme(style=STYLE)
    sns.set_palette(PALETTE)

    if order is None:
        order = list(pd.unique(df[x]))
    if hue and (hue_order is None):
        hue_order = list(pd.unique(df[hue]))

    ax = sns.barplot(data=df, x=x, y=mean_col, hue=hue, order=order, hue_order=hue_order, errorbar=None)

    # Decide y-scale
    mode = _auto_log_decision(df[mean_col].values, yscale)
    if mode == "log":
        ax.set_yscale("log")

    # Overlay error bars
    df_valid = df.dropna(subset=[mean_col]).copy()

    if hue:
        std_lookup = {(str(r[x]), str(r[hue])): r[std_col] for _, r in df_valid[[x, hue, std_col]].iterrows()}
        mean_lookup = {(str(r[x]), str(r[hue])): r[mean_col] for _, r in df_valid[[x, hue, mean_col]].iterrows()}
        present_hues_by_x = df_valid.groupby(x, observed=True)[hue].apply(lambda s: list(pd.unique(s))).to_dict()
        patch_i = 0
        for xv in order:
            present_hues = present_hues_by_x.get(xv, [])
            for hv in hue_order:
                if hv in present_hues:
                    if patch_i >= len(ax.patches):
                        break
                    patch = ax.patches[patch_i]
                    m = mean_lookup.get((str(xv), str(hv)), np.nan)
                    s = std_lookup.get((str(xv), str(hv)), np.nan)
                    lo, up = _asym_err_for_scale(m, s, mode)
                    if pd.notna(lo) and pd.notna(up):
                        xc = patch.get_x() + patch.get_width() / 2
                        yc = patch.get_height()
                        ax.errorbar(
                            xc,
                            yc,
                            yerr=np.array([[lo], [up]]),
                            fmt="none",
                            ecolor=ERROR_COLOR,
                            elinewidth=ERROR_LINEWIDTH,
                            capsize=ERROR_CAPSIZE,
                            zorder=3,
                        )
                    patch_i += 1
    else:
        labels_x = [t.get_text() for t in ax.get_xticklabels()]
        std_lookup = {str(r[x]): r[std_col] for _, r in df_valid[[x, std_col]].iterrows()}
        mean_lookup = {str(r[x]): r[mean_col] for _, r in df_valid[[x, mean_col]].iterrows()}
        for i, patch in enumerate(ax.patches):
            if i >= len(labels_x):
                break
            x_key = labels_x[i]
            m = mean_lookup.get(str(x_key), np.nan)
            s = std_lookup.get(str(x_key), np.nan)
            lo, up = _asym_err_for_scale(m, s, mode)
            if pd.notna(lo) and pd.notna(up):
                xc = patch.get_x() + patch.get_width() / 2
                yc = patch.get_height()
                ax.errorbar(
                    xc,
                    yc,
                    yerr=np.array([[lo], [up]]),
                    fmt="none",
                    ecolor=ERROR_COLOR,
                    elinewidth=ERROR_LINEWIDTH,
                    capsize=ERROR_CAPSIZE,
                    zorder=3,
                )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if hue:
        # Ensure single legend (catplots handled elsewhere; this is barplot)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.get_legend()
            if leg:
                leg.remove()
            ax.legend(handles, labels, title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    plt.tight_layout()
    if out:
        ax.figure.savefig(out, dpi=PLOT_DPI)
        plt.close(ax.figure)


def overall_stats(df, group_col, value_col):
    return df.groupby(group_col, observed=True, as_index=False).agg(mean=(value_col, "mean"), std=(value_col, "std"))


def score_range_plot_normalized(ranges_df, heur_df, out, show_error=True) -> None:
    sns.set_theme(style=STYLE)
    sns.set_palette(PALETTE)

    r = ranges_df.rename(columns={"min": "min_score", "max": "max_score"})
    h = heur_df.merge(r, on="data", how="left")

    denom = (h["max_score"] - h["min_score"]).replace(0, np.nan)
    h["mean_norm"] = (h["mean_best"] - h["min_score"]) / denom
    h["std_norm"] = h["std_best"] / denom
    h["mean_norm"] = h["mean_norm"].clip(0, 1)
    h["std_norm"] = h["std_norm"].clip(lower=0)

    data_list = sorted(r["data"].unique())
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(data_list))))
    y_pos = {d: i for i, d in enumerate(data_list)}

    for _, row in r.iterrows():
        y = y_pos[row["data"]]
        if row["max_score"] == row["min_score"]:
            ax.hlines(y=y, xmin=0.5, xmax=0.5, linewidth=3, alpha=0.7)
        else:
            ax.hlines(y=y, xmin=0.0, xmax=1.0, linewidth=3, alpha=0.7)

    seen = {}
    for _, row in h.iterrows():
        y = y_pos[row["data"]]
        label = HEURISTIC_LABELS.get(row["heuristic"], row["heuristic"])
        x = row["mean_norm"]
        if pd.isna(x):
            continue
        if show_error:
            xerr = row["std_norm"] if pd.notna(row["std_norm"]) else None
            handle = ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3, alpha=0.85, label=label)
        else:
            handle = ax.plot(x, y, "o", alpha=0.9, label=label)[0]
        seen[label] = handle

    ax.set_yticks(list(y_pos.values()), list(y_pos.keys()))
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Best solution score (normalized within each data)")
    suffix = "" if show_error else " [no error]"
    ax.set_title(TITLES["range_plot"] + " [normalized]" + suffix)
    ax.legend(seen.values(), seen.keys(), bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.grid(True, axis="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=PLOT_DPI)
    plt.close(fig)


def score_range_positions_overview(ranges_df, heur_df, out) -> None:
    """Clean scatter overview: one dot per heuristic per data at normalized position."""
    sns.set_theme(style=STYLE)
    sns.set_palette(PALETTE)

    r = ranges_df.rename(columns={"min": "min_score", "max": "max_score"})
    h = heur_df.merge(r, on="data", how="left")
    denom = (h["max_score"] - h["min_score"]).replace(0, np.nan)
    h["pos"] = ((h["mean_best"] - h["min_score"]) / denom).clip(0, 1)

    y_order = sorted(h["data"].unique())
    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.7 * len(y_order))))
    sns.scatterplot(data=h, x="pos", y="data", hue="heuristic", hue_order=HEURISTIC_ORDER, s=70, ax=ax, legend=True)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Position in range (0 = min, 1 = max)")
    ax.set_ylabel("data")
    ax.set_title("Heuristic positions within normalized score ranges (all data)")
    ax.grid(True, axis="both", alpha=0.2)
    # Ensure single legend
    leg = ax.get_legend()
    if leg:
        leg.remove()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Heuristic", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=PLOT_DPI)
    plt.close(fig)


def score_boxplot_by_k_heuristic(df_runs: pd.DataFrame, out, yscale="auto") -> None:
    """
    Box plot of raw best scores by number of clusters (k) and heuristic.
    Uses all runs and datasets; hide outliers for cleaner view.
    """
    sns.set_theme(style=STYLE)
    sns.set_palette(PALETTE)
    d = df_runs.dropna(subset=["best_score"]).copy()
    g = sns.catplot(
        data=d,
        kind="box",
        x="k",
        y="best_score",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        showfliers=False,
        height=5,
        aspect=1.6,
        legend=True,
    )
    # Remove auto legend and add one
    if getattr(g, "_legend", None) is not None:
        g._legend.remove()
    ax = g.axes[0, 0]
    mode = _auto_log_decision(d["best_score"].values, yscale)
    if mode == "log" and np.nanmin(d["best_score"].values) > 0:
        ax.set_yscale("log")
    ax.set_title(TITLES["score_box_k_heur"])
    ax.set_xlabel(XTEXT["k"])
    ax.set_ylabel(YTEXT["score"])
    ax.legend(title=XTEXT["heuristic"], bbox_to_anchor=(1.02, 1), loc="upper left")
    g.fig.tight_layout()
    g.fig.savefig(out, dpi=PLOT_DPI)
    plt.close(g.fig)


def score_boxplot_per_data(df_runs: pd.DataFrame, outdir: Path, yscale="auto") -> None:
    """One box-plot per dataset: best score by k and heuristic."""
    sns.set_theme(style=STYLE)
    sns.set_palette(PALETTE)
    for data_name, d in df_runs.dropna(subset=["best_score"]).groupby("data"):
        g = sns.catplot(
            data=d,
            kind="box",
            x="k",
            y="best_score",
            hue="heuristic",
            hue_order=HEURISTIC_ORDER,
            showfliers=False,
            height=5,
            aspect=1.6,
            legend=True,
        )
        if getattr(g, "_legend", None) is not None:
            g._legend.remove()
        ax = g.axes[0, 0]
        mode = _auto_log_decision(d["best_score"].values, yscale)
        if mode == "log" and np.nanmin(d["best_score"].values) > 0:
            ax.set_yscale("log")
        ax.set_title(TITLES["score_box_k_heur_per_data"].format(data_name))
        ax.set_xlabel(XTEXT["k"])
        ax.set_ylabel(YTEXT["score"])
        ax.legend(title=XTEXT["heuristic"], bbox_to_anchor=(1.02, 1), loc="upper left")
        out = outdir / f"score_box_by_k_and_heuristic_{data_name}.png"
        g.fig.tight_layout()
        g.fig.savefig(out, dpi=PLOT_DPI)
        plt.close(g.fig)


# =========================
# Main
# =========================
def main(indir: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style=STYLE)

    df = gather_records(indir)
    if df.empty:
        print("No .log files found. Check --indir.")
        return
    df.to_csv(outdir / "parsed_runs.csv", index=False)

    agg = aggregate_runs(df)
    agg["heuristic_label"] = agg["heuristic"].astype(str).replace(HEURISTIC_LABELS)

    # -------- CPU --------
    cpu_overall = overall_stats(agg, "heuristic", "cpu_time_mean")
    cpu_order = cpu_overall.sort_values("mean")["heuristic"].tolist()

    cpu_hd = agg.groupby(["heuristic", "data"], observed=True, as_index=False).agg(
        mean=("cpu_time_mean", "mean"), std=("cpu_time_mean", "std")
    )
    bar_with_std(
        cpu_hd,
        x="heuristic",
        mean_col="mean",
        std_col="std",
        hue="data",
        order=cpu_order,
        title=TITLES["cpu_by_heur_data"],
        xlabel=XTEXT["heuristic"],
        ylabel=YTEXT["cpu"],
        out=outdir / "cpu_by_heuristic_and_data.png",
        yscale="auto",
    )

    bar_with_std(
        cpu_overall.sort_values("mean"),
        x="heuristic",
        mean_col="mean",
        std_col="std",
        order=cpu_order,
        title=TITLES["cpu_by_heur_overall"],
        xlabel=XTEXT["heuristic"],
        ylabel=YTEXT["cpu"],
        out=outdir / "cpu_by_heuristic_overall.png",
        yscale="auto",
    )

    cpu_kh = agg.groupby(["k", "heuristic"], observed=True, as_index=False).agg(
        mean=("cpu_time_mean", "mean"), std=("cpu_time_mean", "std")
    )
    bar_with_std(
        cpu_kh,
        x="k",
        mean_col="mean",
        std_col="std",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        title=TITLES["cpu_by_k_heur"],
        xlabel=XTEXT["k"],
        ylabel=YTEXT["cpu"],
        out=outdir / "cpu_by_k_and_heuristic.png",
        yscale="auto",
    )

    cpu_Dh = agg.groupby(["D", "heuristic"], observed=True, as_index=False).agg(
        mean=("cpu_time_mean", "mean"), std=("cpu_time_mean", "std")
    )
    bar_with_std(
        cpu_Dh,
        x="D",
        mean_col="mean",
        std_col="std",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        title=TITLES["cpu_by_D_heur"],
        xlabel=XTEXT["D"],
        ylabel=YTEXT["cpu"],
        out=outdir / "cpu_by_D_and_heuristic.png",
        yscale="auto",
    )

    cpu_nh = agg.groupby(["n", "heuristic"], observed=True, as_index=False).agg(
        mean=("cpu_time_mean", "mean"), std=("cpu_time_mean", "std")
    )
    bar_with_std(
        cpu_nh,
        x="n",
        mean_col="mean",
        std_col="std",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        title=TITLES["cpu_by_n_heur"],
        xlabel=XTEXT["n"],
        ylabel=YTEXT["cpu"],
        out=outdir / "cpu_by_n_and_heuristic.png",
        yscale="auto",
    )

    # -------- Wall --------
    wall_overall = overall_stats(agg, "heuristic", "wall_time_mean")
    wall_order = wall_overall.sort_values("mean")["heuristic"].tolist()

    wall_hd = agg.groupby(["heuristic", "data"], observed=True, as_index=False).agg(
        mean=("wall_time_mean", "mean"), std=("wall_time_mean", "std")
    )
    bar_with_std(
        wall_hd,
        x="heuristic",
        mean_col="mean",
        std_col="std",
        hue="data",
        order=wall_order,
        title=TITLES["wall_by_heur_data"],
        xlabel=XTEXT["heuristic"],
        ylabel=YTEXT["wall"],
        out=outdir / "wall_by_heuristic_and_data.png",
        yscale="auto",
    )

    bar_with_std(
        wall_overall.sort_values("mean"),
        x="heuristic",
        mean_col="mean",
        std_col="std",
        order=wall_order,
        title=TITLES["wall_by_heur_overall"],
        xlabel=XTEXT["heuristic"],
        ylabel=YTEXT["wall"],
        out=outdir / "wall_by_heuristic_overall.png",
        yscale="auto",
    )

    wall_kh = agg.groupby(["k", "heuristic"], observed=True, as_index=False).agg(
        mean=("wall_time_mean", "mean"), std=("wall_time_mean", "std")
    )
    bar_with_std(
        wall_kh,
        x="k",
        mean_col="mean",
        std_col="std",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        title=TITLES["wall_by_k_heur"],
        xlabel=XTEXT["k"],
        ylabel=YTEXT["wall"],
        out=outdir / "wall_by_k_and_heuristic.png",
        yscale="auto",
    )

    wall_Dh = agg.groupby(["D", "heuristic"], observed=True, as_index=False).agg(
        mean=("wall_time_mean", "mean"), std=("wall_time_mean", "std")
    )
    bar_with_std(
        wall_Dh,
        x="D",
        mean_col="mean",
        std_col="std",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        title=TITLES["wall_by_D_heur"],
        xlabel=XTEXT["D"],
        ylabel=YTEXT["wall"],
        out=outdir / "wall_by_D_and_heuristic.png",
        yscale="auto",
    )

    wall_nh = agg.groupby(["n", "heuristic"], observed=True, as_index=False).agg(
        mean=("wall_time_mean", "mean"), std=("wall_time_mean", "std")
    )
    bar_with_std(
        wall_nh,
        x="n",
        mean_col="mean",
        std_col="std",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        title=TITLES["wall_by_n_heur"],
        xlabel=XTEXT["n"],
        ylabel=YTEXT["wall"],
        out=outdir / "wall_by_n_and_heuristic.png",
        yscale="auto",
    )

    # -------- Best score (bar charts) --------
    score_hd = agg.groupby(["heuristic", "data"], observed=True, as_index=False).agg(
        mean=("best_score_mean", "mean"), std=("best_score_mean", "std")
    )
    bar_with_std(
        score_hd,
        x="heuristic",
        mean_col="mean",
        std_col="std",
        hue="data",
        order=HEURISTIC_ORDER,
        title=TITLES["score_by_heur_data"],
        xlabel=XTEXT["heuristic"],
        ylabel=YTEXT["score"],
        out=outdir / "score_by_heuristic_and_data.png",
        yscale="auto",
    )

    score_overall = overall_stats(agg, "heuristic", "best_score_mean")
    bar_with_std(
        score_overall,
        x="heuristic",
        mean_col="mean",
        std_col="std",
        order=HEURISTIC_ORDER,
        title=TITLES["score_by_heur_overall"],
        xlabel=XTEXT["heuristic"],
        ylabel=YTEXT["score"],
        out=outdir / "score_by_heuristic_overall.png",
        yscale="auto",
    )

    score_kh = agg.groupby(["k", "heuristic"], observed=True, as_index=False).agg(
        mean=("best_score_mean", "mean"), std=("best_score_mean", "std")
    )
    bar_with_std(
        score_kh,
        x="k",
        mean_col="mean",
        std_col="std",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        title=TITLES["score_by_k_heur"],
        xlabel=XTEXT["k"],
        ylabel=YTEXT["score"],
        out=outdir / "score_by_k_and_heuristic.png",
        yscale="auto",
    )

    score_Dh = agg.groupby(["D", "heuristic"], observed=True, as_index=False).agg(
        mean=("best_score_mean", "mean"), std=("best_score_mean", "std")
    )
    bar_with_std(
        score_Dh,
        x="D",
        mean_col="mean",
        std_col="std",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        title=TITLES["score_by_D_heur"],
        xlabel=XTEXT["D"],
        ylabel=YTEXT["score"],
        out=outdir / "score_by_D_and_heuristic.png",
        yscale="auto",
    )

    score_nh = agg.groupby(["n", "heuristic"], observed=True, as_index=False).agg(
        mean=("best_score_mean", "mean"), std=("best_score_mean", "std")
    )
    bar_with_std(
        score_nh,
        x="n",
        mean_col="mean",
        std_col="std",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        title=TITLES["score_by_n_heur"],
        xlabel=XTEXT["n"],
        ylabel=YTEXT["score"],
        out=outdir / "score_by_n_and_heuristic.png",
        yscale="auto",
    )

    # -------- Box plots: all data & per data --------
    score_boxplot_by_k_heuristic(df_runs=df, out=outdir / "score_box_by_k_and_heuristic.png", yscale="auto")
    score_boxplot_per_data(df_runs=df, outdir=outdir, yscale="auto")

    # -------- Score ranges (normalized) --------
    ranges_df, heur_df = compute_score_range_by_data(df)
    score_range_plot_normalized(ranges_df, heur_df, out=outdir / "score_range_by_data_normalized.png", show_error=True)
    score_range_plot_normalized(
        ranges_df, heur_df, out=outdir / "score_range_by_data_normalized_noerr.png", show_error=False
    )
    score_range_positions_overview(ranges_df, heur_df, out=outdir / "scoreRange_norm_overview.png")
    ranges_df.to_csv(outdir / "score_ranges_by_data.csv", index=False)
    heur_df.to_csv(outdir / "heuristic_positions_by_data.csv", index=False)

    # -------- ARI --------
    per_k_ari, by_data_heur, by_heur = compute_mean_ari(df)
    per_k_ari.to_csv(outdir / "ari_by_data_heur_k.csv", index=False)
    by_data_heur.to_csv(outdir / "ari_by_data_heur.csv", index=False)
    by_heur.to_csv(outdir / "ari_by_heuristic.csv", index=False)

    # ARI by data & heuristic — ensure single legend (remove catplot's, then add ours)
    sns.set_theme(style=STYLE)
    df1 = by_data_heur.copy()
    df1["heuristic"] = pd.Categorical(df1["heuristic"], categories=HEURISTIC_ORDER, ordered=True)
    g1 = sns.catplot(
        data=df1,
        kind="bar",
        x="data",
        y="mean_ari",
        hue="heuristic",
        hue_order=HEURISTIC_ORDER,
        errorbar=None,
        height=5,
        aspect=1.6,
        legend=True,
    )
    if getattr(g1, "_legend", None) is not None:
        g1._legend.remove()
    ax1 = g1.axes[0, 0]
    ax1.set_title(TITLES["ari_by_data_heur"])
    ax1.set_xlabel(XTEXT["data"])
    ax1.set_ylabel(YTEXT["ari"])
    ax1.legend(title=XTEXT["heuristic"], bbox_to_anchor=(1.02, 1), loc="upper left")
    g1.fig.tight_layout()
    g1.fig.savefig(outdir / "ari_by_data_and_heuristic.png", dpi=PLOT_DPI)
    plt.close(g1.fig)

    # ARI by heuristic overall (mean ± std) — keep linear scale
    df2 = by_heur.copy()
    df2["heuristic"] = pd.Categorical(df2["heuristic"], categories=HEURISTIC_ORDER, ordered=True)
    tmp = df2.rename(columns={"mean_ari": "mean", "std_ari": "std"})
    bar_with_std(
        tmp,
        x="heuristic",
        mean_col="mean",
        std_col="std",
        order=HEURISTIC_ORDER,
        title=TITLES["ari_by_heur"],
        xlabel=XTEXT["heuristic"],
        ylabel=YTEXT["ari"],
        out=outdir / "ari_by_heuristic_overall.png",
        yscale="linear",
    )

    print(f"Done. Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, required=True, help="Directory with *.log files")
    ap.add_argument("--outdir", type=Path, default=Path("./plots"), help="Output directory for figures/CSVs")
    args = ap.parse_args()
    main(args.indir, args.outdir)
