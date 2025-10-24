#!/usr/bin/env python3

import argparse
import contextlib
import glob
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

FNAME_RE = re.compile(r"^(?P<data>.+)_(?P<heur>(lp|ls|rd|sls))_(?P<k>\d+)_(?P<run>\d+)\.log$")
RX_D_AND_N = re.compile(r"\|D\|\s*=\s*(?P<D>\d+).*?n\s*=\s*(?P<n>\d+)", re.IGNORECASE)
RX_CPU = re.compile(r"^\[CPU time\]\s+(?P<sec>[0-9.]+)\s+seconds")
RX_WALL = re.compile(r"^\[Wall time\]\s+(?P<sec>[0-9.]+)\s+seconds")
RX_BEST = re.compile(r"^\[best solution\]:\s*(?P<clusters>\[.*?\])\s*\|\s*(?P<score>[-+eE0-9.]+)")
RX_TRAIL_SCORE = re.compile(r"\|\s*([-+eE0-9.]+)\s*$")
RX_BRACKETS = re.compile(r"\[(.*?)\]")

HEUR_ORDER = ["lp", "ls", "rd", "sls"]


def parse_filename(path: str) -> Dict[str, Any]:
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        msg = f"Filename does not match expected pattern: {path}"
        raise ValueError(msg)
    gd = m.groupdict()
    return {"data": gd["data"], "heuristic": gd["heur"], "k": int(gd["k"]), "run": int(gd["run"]), "path": path}


def parse_clusters(text: str) -> List[int]:
    m = RX_BRACKETS.search(text)
    if not m:
        return []
    inside = m.group(1).strip()
    if not inside:
        return []
    parts = [p.strip() for p in inside.replace(",", ";").split(";") if p.strip() != ""]
    try:
        return [int(p) for p in parts]
    except ValueError:
        return [int(float(p)) for p in parts]


def parse_one_log(meta: Dict[str, Any]) -> Dict[str, Any]:
    D = n = None
    cpu = wall = best_score = None
    clusters = None
    trailing_scores: List[float] = []
    with open(meta["path"], encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            mx = RX_D_AND_N.search(line)
            if mx:
                D, n = int(mx.group("D")), int(mx.group("n"))
            mx = RX_CPU.match(line)
            if mx:
                cpu = float(mx.group("sec"))
            mx = RX_WALL.match(line)
            if mx:
                wall = float(mx.group("sec"))
            mx = RX_BEST.match(line)
            if mx:
                clusters = parse_clusters(mx.group("clusters"))
                best_score = float(mx.group("score"))
            mx = RX_TRAIL_SCORE.search(line)
            if mx:
                with contextlib.suppress(ValueError):
                    trailing_scores.append(float(mx.group(1)))
    if D is None or n is None:
        with open(meta["path"], encoding="utf-8", errors="ignore") as f:
            text = f.read()
        md = re.search(r"\|D\|\s*=\s*(\d+)", text, flags=re.IGNORECASE)
        mn = re.search(r"n\s*=\s*(\d+)", text, flags=re.IGNORECASE)
        if md:
            D = int(md.group(1))
        if mn:
            n = int(mn.group(1))
    return {
        **meta,
        "D": D,
        "n": n,
        "cpu_time": cpu,
        "wall_time": wall,
        "best_score": best_score,
        "clusters": clusters,
        "trailing_scores": trailing_scores,
    }


def build_runs_df(log_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, range_rows = [], []
    paths = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not paths:
        msg = f"No .log files found under: {log_dir}"
        raise SystemExit(msg)
    for p in paths:
        try:
            meta = parse_filename(p)
        except ValueError:
            continue
        rec = parse_one_log(meta)
        rows.append(
            {
                "data": rec["data"],
                "heuristic": rec["heuristic"],
                "k": rec["k"],
                "run": rec["run"],
                "D": rec["D"],
                "n": rec["n"],
                "cpu_time": rec["cpu_time"],
                "wall_time": rec["wall_time"],
                "best_score": rec["best_score"],
                "clusters": rec["clusters"],
                "path": rec["path"],
            }
        )
        for s in rec["trailing_scores"]:
            range_rows.append(
                {
                    "data": rec["data"],
                    "heuristic": rec["heuristic"],
                    "k": rec["k"],
                    "run": rec["run"],
                    "score": s,
                    "path": rec["path"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(range_rows)


def compute_aggregates(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Mean + std over runs for cpu_time, wall_time, best_score grouped by data, heuristic, k, D, n."""
    num_cols = ["cpu_time", "wall_time", "best_score"]
    agg = (
        df_runs.groupby(["data", "heuristic", "k", "D", "n"], dropna=False)[num_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # SAFE flatten: leave simple columns (like 'data') untouched; join tuples for multi-index metrics
    new_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            new_cols.append("_".join([part for part in c if part]))
        else:
            new_cols.append(c)
    agg.columns = new_cols

    # keep a stable order for heuristic
    agg["heuristic"] = pd.Categorical(agg["heuristic"], categories=["lp", "ls", "rd", "sls"], ordered=True)
    return agg.sort_values(["data", "k", "heuristic"])


def compute_score_range(df_range: pd.DataFrame, df_runs: pd.DataFrame) -> pd.DataFrame:
    if df_range.empty:
        return pd.DataFrame(columns=["data", "score_min", "score_max", "heuristic", "heur_best_mean", "position_0_1"])
    per_data = (
        df_range.groupby("data")["score"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "score_min", "max": "score_max"})
    )
    best_means = (
        df_runs.groupby(["data", "heuristic"])["best_score"]
        .mean()
        .reset_index()
        .rename(columns={"best_score": "heur_best_mean"})
    )
    out = best_means.merge(per_data, on="data", how="left")
    out["position_0_1"] = (out["heur_best_mean"] - out["score_min"]) / (
        (out["score_max"] - out["score_min"]).replace(0, np.nan)
    )
    out["heuristic"] = pd.Categorical(out["heuristic"], categories=HEUR_ORDER, ordered=True)
    return out.sort_values(["data", "heuristic"])


def compute_mean_ari(df_runs: pd.DataFrame) -> pd.DataFrame:
    """For each (data, k, heuristic), compute mean and std of pairwise Adjusted Rand Index across runs."""
    sub = df_runs.dropna(subset=["clusters"]).copy()
    rows = []
    for (data, k, heur), g in sub.groupby(["data", "k", "heuristic"]):
        items = list(g.itertuples(index=False))
        if len(items) < 2:
            continue
        aris = []
        for r1, r2 in combinations(items, 2):
            c1, c2 = r1.clusters, r2.clusters
            if c1 is None or c2 is None:
                continue
            if len(c1) != len(c2):
                continue
            with contextlib.suppress(Exception):
                aris.append(adjusted_rand_score(c1, c2))
        if aris:
            rows.append(
                {
                    "data": data,
                    "k": k,
                    "heuristic": heur,
                    "mean_ARI": float(np.mean(aris)),
                    "std_ARI": float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0,
                    "pairs_used": len(aris),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["heuristic"] = pd.Categorical(df["heuristic"], categories=HEUR_ORDER, ordered=True)
        df = df.sort_values(["data", "k", "heuristic"])
    return df


# ---------- Plot helpers: bars (mean±std) and box plots from run-level ----------


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def bar_mean_std(df: pd.DataFrame, x: str, hue: str, metric: str, title: str, xlabel: str, ylabel: str, out: Path) -> None:
    tmp = df.rename(columns={f"{metric}_mean": "mean", f"{metric}_std": "std"}).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=tmp, x=x, y="mean", hue=hue, errorbar=None, ax=ax, dodge=True)
    # manually draw ±std error bars per (x,hue)
    if hue:
        for (xi, hi), sub in tmp.groupby([x, hue]):
            m = sub["mean"].mean()
            s = sub["std"].mean()
            # place error bar at the category position; simplest: overlay scatter+error
            ax.errorbar(x=xi, y=m, yerr=0.0 if pd.isna(s) else s, fmt="none", capsize=3)
    else:
        for xi, sub in tmp.groupby(x):
            m = sub["mean"].mean()
            s = sub["std"].mean()
            ax.errorbar(x=xi, y=m, yerr=0.0 if pd.isna(s) else s, fmt="none", capsize=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    savefig(fig, out)


def box_from_runs(
    df_runs: pd.DataFrame, x: str, hue: str, metric: str, title: str, xlabel: str, ylabel: str, out: Path
) -> None:
    """Box plot from raw per-run values (more informative distributions)."""
    sub = df_runs.dropna(subset=[metric]).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=sub, x=x, y=metric, hue=hue, ax=ax, dodge=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    savefig(fig, out)


def _barplot_sd(df, x, y, hue, ax) -> None:
    """Bar plot of mean with SD error bars, compatible with seaborn 0.11 and 0.12+."""
    try:
        sns.barplot(data=df, x=x, y=y, hue=hue, estimator=np.mean, errorbar="sd", dodge=True, ax=ax)
    except TypeError:
        # seaborn<0.12
        sns.barplot(data=df, x=x, y=y, hue=hue, estimator=np.mean, ci="sd", dodge=True, ax=ax)


def _savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_views(df_runs: pd.DataFrame, agg: pd.DataFrame, outdir: Path, metric: str) -> None:
    label = {"cpu_time": "CPU time (s)", "wall_time": "Wall time (s)", "best_score": "Best score"}[metric]
    ensure_dir(outdir)

    # --- (1) by heuristic and data: one figure per data (x=k, hue=heuristic), bars = mean±sd over runs ---
    for data_val, sub in df_runs[df_runs[metric].notna()].groupby("data"):
        fig, ax = plt.subplots(figsize=(10, 6))
        _barplot_sd(sub, x="k", y=metric, hue="heuristic", ax=ax)
        ax.set_title(f"{label} — by k and heuristic (data={data_val})")
        ax.set_xlabel("k (clusters)")
        ax.set_ylabel(label)
        _savefig(fig, outdir / f"{metric}_bar_byKHeuristic_data_{data_val}.png")

        # Optional: a box plot to show distribution over runs
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=sub, x="k", y=metric, hue="heuristic", dodge=True, ax=ax)
        ax.set_title(f"{label} — distribution by k and heuristic (box, data={data_val})")
        ax.set_xlabel("k (clusters)")
        ax.set_ylabel(label)
        _savefig(fig, outdir / f"{metric}_box_byKHeuristic_data_{data_val}.png")

    # --- (2) by heuristic across all data: bars = mean±sd (pooled across data, k, runs) ---
    sub = df_runs[df_runs[metric].notna()].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    _barplot_sd(sub, x="heuristic", y=metric, hue=None, ax=ax)
    ax.set_title(f"{label} — by heuristic (across all data)")
    ax.set_xlabel("Heuristic")
    ax.set_ylabel(label)
    _savefig(fig, outdir / f"{metric}_bar_byHeuristic_overall.png")

    # --- (3) by number of clusters and heuristic (pooled across data) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    _barplot_sd(sub, x="k", y=metric, hue="heuristic", ax=ax)
    ax.set_title(f"{label} — by k and heuristic (all data pooled)")
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel(label)
    _savefig(fig, outdir / f"{metric}_bar_byKHeuristic.png")

    # --- (4) by number of indicators |D| and heuristic (pooled) ---
    subD = sub.dropna(subset=["D"])
    if not subD.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        _barplot_sd(subD, x="D", y=metric, hue="heuristic", ax=ax)
        ax.set_title(f"{label} — by |D| and heuristic (all data pooled)")
        ax.set_xlabel("|D| (number of indicators)")
        ax.set_ylabel(label)
        _savefig(fig, outdir / f"{metric}_bar_byDHeuristic.png")

    # --- (5) by size n and heuristic (pooled) ---
    subN = sub.dropna(subset=["n"])
    if not subN.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        _barplot_sd(subN, x="n", y=metric, hue="heuristic", ax=ax)
        ax.set_title(f"{label} — by n and heuristic (all data pooled)")
        ax.set_xlabel("n (data size)")
        ax.set_ylabel(label)
        _savefig(fig, outdir / f"{metric}_bar_byNHeuristic.png")


# --- helper: put errorbars exactly over each bar (in draw order) ---
def _errorbars_over_bars(ax, yerrs) -> None:
    """
    Add vertical error bars centered on each bar in `ax`, matching the draw order
    of the bars. `yerrs` must have the same length as the number of bars.
    """
    bars = [p for p in ax.patches if hasattr(p, "get_width")]  # all drawn bars
    if len(bars) != len(yerrs):
        # If lengths ever mismatch, align by the min length to avoid crashes
        n = min(len(bars), len(yerrs))
        bars = bars[:n]
        yerrs = yerrs[:n]
    for p, e in zip(bars, yerrs):
        if pd.isna(e) or e is None:
            e = 0.0
        x = p.get_x() + p.get_width() / 2.0
        y = p.get_height()
        ax.errorbar(x, y, yerr=e, fmt="none", capsize=4, color="black", lw=1, zorder=5)


def plot_mean_ari(df_mean_ari: pd.DataFrame, outdir: Path) -> None:
    """
    Bars with correctly-aligned ±std error bars:
      - Overall (pooled across data): for each (k, heuristic), bar = mean of mean_ARI across data,
        error = std of mean_ARI across data.
      - Per data: for each (k, heuristic), bar = mean_ARI, error = std_ARI (within-data pairwise std).
    """
    if df_mean_ari.empty:
        return

    # consistent ordering
    k_order = sorted(df_mean_ari["k"].unique().tolist())
    heur_order = HEUR_ORDER
    df_mean_ari = df_mean_ari.copy()
    df_mean_ari["heuristic"] = pd.Categorical(df_mean_ari["heuristic"], categories=heur_order, ordered=True)

    # ===== Overall across data =====
    overall = (
        df_mean_ari.groupby(["k", "heuristic"], as_index=False)
        .agg(mean_ARI=("mean_ARI", "mean"), std_over_data=("mean_ARI", "std"))
        .sort_values(["k", "heuristic"])
    )
    overall["k"] = pd.Categorical(overall["k"], categories=k_order, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    # draw bars (no errorbar here)
    sns.barplot(data=overall, x="k", y="mean_ARI", hue="heuristic", dodge=True, errorbar=None, ax=ax)
    # add error bars exactly over bars, in draw order
    _errorbars_over_bars(ax, overall["std_over_data"].fillna(0.0).tolist())

    ax.set_title("Mean ARI by k and heuristic (pooled across data)")
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel("Mean ARI ± std (across data)")
    savefig(fig, outdir / "meanARI_bar_byKHeuristic_overall.png")

    # ===== Per-data small multiples =====
    for data_val, g in df_mean_ari.groupby("data"):
        gg = g.sort_values(["k", "heuristic"]).copy()
        gg["k"] = pd.Categorical(gg["k"], categories=k_order, ordered=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=gg, x="k", y="mean_ARI", hue="heuristic", dodge=True, errorbar=None, ax=ax)
        # here the error is the within-data pairwise std we computed earlier
        _errorbars_over_bars(ax, gg["std_ARI"].fillna(0.0).tolist())

        ax.set_title(f"Mean ARI by k and heuristic — data={data_val}")
        ax.set_xlabel("k (clusters)")
        ax.set_ylabel("Mean ARI ± std (pairwise within data)")
        savefig(fig, outdir / f"meanARI_bar_byKHeuristic_{data_val}.png")


def plot_score_range_report(range_report: pd.DataFrame, outdir: Path) -> None:
    """
    Visualize, for each 'data', the overall score range [min, max] and where each heuristic's
    mean best score sits inside that range. Saves:
      - scoreRange_abs_{data}.png  (absolute scores)
      - scoreRange_norm_{data}.png (normalized [0,1] positions)
      - scoreRange_norm_overview.png (all data stacked, normalized).
    """
    if range_report is None or range_report.empty:
        return

    ensure_dir(outdir)

    # Make sure heuristics have a stable plotting order
    heur_order = ["lp", "ls", "rd", "sls"]
    rr = range_report.copy()
    rr["heuristic"] = pd.Categorical(rr["heuristic"], categories=heur_order, ordered=True)

    # Small vertical offsets to avoid marker overlap on a single baseline
    offsets = dict(zip(heur_order, [-0.30, -0.10, 0.10, 0.30]))

    # --- Per-data plots ---
    for data_val, g in rr.groupby("data", sort=False):
        g = g.sort_values("heuristic")
        score_min = float(g["score_min"].iloc[0])
        score_max = float(g["score_max"].iloc[0])

        # ===== Absolute scale =====
        fig, ax = plt.subplots(figsize=(10, 3.8))
        # baseline segment [min, max]
        ax.hlines(y=0, xmin=score_min, xmax=score_max, linewidth=3)
        # add heuristic points with small vertical offsets + labels
        pad = (score_max - score_min) * 0.05 if score_max > score_min else 1.0
        for row in g.itertuples(index=False):
            y = offsets.get(row.heuristic, 0.0)
            ax.scatter(row.heur_best_mean, y, s=60)
            ax.text(row.heur_best_mean, y + 0.05, str(row.heuristic), ha="center", va="bottom", fontsize=9)

        ax.set_title(f"Score range and heuristic positions — data={data_val}")
        ax.set_xlabel("Score (absolute)")
        ax.set_yticks([])  # cosmetic
        ax.set_xlim(score_min - pad, score_max + pad)
        _savefig(fig, outdir / f"scoreRange_abs_{data_val}.png")

        # ===== Normalized [0,1] =====
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.hlines(y=0, xmin=0.0, xmax=1.0, linewidth=3)
        for row in g.itertuples(index=False):
            y = offsets.get(row.heuristic, 0.0)
            x = row.position_0_1
            ax.scatter(x, y, s=60)
            ax.text(x, y + 0.05, str(row.heuristic), ha="center", va="bottom", fontsize=9)

        ax.set_title(f"Normalized score range [0–1] — data={data_val}")
        ax.set_xlabel("Position in range (0 = min, 1 = max)")
        ax.set_yticks([])
        ax.set_xlim(-0.05, 1.05)
        _savefig(fig, outdir / f"scoreRange_norm_{data_val}.png")

    # --- Overview across all data in normalized space ---
    # One strip/swarm-like chart: y=data, x=position_0_1, hue=heuristic
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * rr["data"].nunique() + 2)))
    # Use stripplot (no dodge) to keep it clean; points carry heuristic hue
    sns.stripplot(data=rr, x="position_0_1", y="data", hue="heuristic", ax=ax, jitter=False, dodge=True)
    # Draw faint baseline [0,1] for each row
    for yi, data_cat in enumerate(ax.get_yticklabels()):
        ax.hlines(y=yi, xmin=0.0, xmax=1.0, linewidth=2, alpha=0.25, zorder=0)
    ax.set_title("Heuristic positions within normalized score ranges (all data)")
    ax.set_xlabel("Position in range (0 = min, 1 = max)")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(title="Heuristic", bbox_to_anchor=(1.02, 1), loc="upper left")
    _savefig(fig, outdir / "scoreRange_norm_overview.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse experiment logs and plot bar/box charts with Seaborn.")
    parser.add_argument("log_dir", help="Directory with *.log named data_heuristic_k_run.log")
    parser.add_argument("--out", default="out", help="Output directory for plots and reports")
    args = parser.parse_args()

    out_dir = Path(args.out)
    plots_dir = out_dir / "plots"
    reps_dir = out_dir / "reports"
    ensure_dir(plots_dir)
    ensure_dir(reps_dir)

    df_runs, df_range = build_runs_df(args.log_dir)
    df_runs.to_csv(reps_dir / "runs_parsed.csv", index=False)
    df_range.to_csv(reps_dir / "trailing_scores_raw.csv", index=False)

    agg = compute_aggregates(df_runs)
    agg.to_csv(reps_dir / "aggregated_mean_std.csv", index=False)

    range_report = compute_score_range(df_range, df_runs)
    range_report.to_csv(reps_dir / "score_range_report_by_data.csv", index=False)

    plot_score_range_report(range_report, plots_dir)

    df_mean_ari = compute_mean_ari(df_runs)
    df_mean_ari.to_csv(reps_dir / "mean_ari_by_data_k_heuristic.csv", index=False)

    # Plot CPU / Wall / Best — bars + selected boxes
    for metric in ["cpu_time", "wall_time", "best_score"]:
        plot_metric_views(df_runs, agg, plots_dir, metric)

    # Plot ARI bars
    plot_mean_ari(df_mean_ari, plots_dir)

    print("Saved:")
    print(f"- runs_parsed.csv -> {reps_dir / 'runs_parsed.csv'}")
    print(f"- aggregated_mean_std.csv -> {reps_dir / 'aggregated_mean_std.csv'}")
    print(f"- score_range_report_by_data.csv -> {reps_dir / 'score_range_report_by_data.csv'}")
    print(f"- mean_ari_by_data_k_heuristic.csv -> {reps_dir / 'mean_ari_by_data_k_heuristic.csv'}")
    print(f"- plots -> {plots_dir}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
