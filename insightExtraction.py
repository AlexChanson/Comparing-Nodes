import re
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt



def build_feature_ranges_from_beforeValidation(
    beforeValidation,
    feature_names,
    *,
    use_percentiles=False,
    p_low=1.0,
    p_high=99.0,
    widen_if_constant=1.0
):
    """
    Build per-feature (min,max) ranges from original data in beforeValidation.

    Assumes:
      - beforeValidation[:,0] is outid
      - beforeValidation[:,1:] are the original feature values aligned with feature_names

    Parameters
    ----------
    use_percentiles : bool
        If True, use robust percentile bounds instead of min/max to avoid outliers.
    p_low, p_high : float
        Percentiles used if use_percentiles=True.
    widen_if_constant : float
        If a feature is constant (min==max), expand range to [v-widen, v+widen].

    Returns
    -------
    dict[str, tuple(float,float)] mapping feature name -> (min,max)
    """
    bv = np.asarray(beforeValidation, dtype=float)
    Xorig = bv[:, 1:]  # drop outid
    feature_names = list(feature_names)

    if Xorig.shape[1] != len(feature_names):
        raise ValueError(
            f"beforeValidation has {Xorig.shape[1]} feature columns but feature_names has {len(feature_names)}"
        )

    ranges = {}
    for j, fname in enumerate(feature_names):
        col = Xorig[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            # fallback: arbitrary range so plotting doesn't crash
            ranges[fname] = (0.0, 1.0)
            continue

        if use_percentiles:
            mn = float(np.percentile(col, p_low))
            mx = float(np.percentile(col, p_high))
        else:
            mn = float(np.min(col))
            mx = float(np.max(col))

        if not np.isfinite(mn) or not np.isfinite(mx):
            ranges[fname] = (0.0, 1.0)
            continue

        if mx <= mn:
            v = mn
            ranges[fname] = (v - widen_if_constant, v + widen_if_constant)
        else:
            ranges[fname] = (mn, mx)

    return ranges


# ---------------------------
# Radar chart helpers
# ---------------------------
def plot_pair_radar(
    feature_names,
    a_values,
    b_values,
    *,
    feature_ranges,          # REQUIRED: dict feature -> (min,max)
    a_label="A",
    b_label="B",
    title=None,
    outpath=None,
    color_a="tab:blue",
    color_b="tab:orange",
    annotate_values=True,
    clip=True
):
    """
    Plot A vs B on a radar chart using per-feature ranges, scaling each feature to [0,1].

    Radar charts have a shared radial axis; per-feature ranges are applied by scaling
    each dimension independently to [0,1] before plotting.

    feature_ranges must come from beforeValidation (original values).
    """
    feats = list(feature_names)
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)

    if len(feats) == 0:
        return
    if a.shape[0] != len(feats) or b.shape[0] != len(feats):
        raise ValueError("plot_pair_radar: feature_names and values must have the same length.")

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    # Build per-feature ranges list aligned with feats
    ranges = []
    for f in feats:
        if f not in feature_ranges:
            raise KeyError(f"Missing range for feature {f!r}. Build ranges from beforeValidation first.")
        mn, mx = feature_ranges[f]
        if mx <= mn:
            mx = mn + 1.0
        ranges.append((float(mn), float(mx)))

    # Scale to [0,1]
    a_scaled = np.empty_like(a, dtype=float)
    b_scaled = np.empty_like(b, dtype=float)
    for i, (mn, mx) in enumerate(ranges):
        a_scaled[i] = (a[i] - mn) / (mx - mn)
        b_scaled[i] = (b[i] - mn) / (mx - mn)

    if clip:
        a_scaled = np.clip(a_scaled, 0.0, 1.0)
        b_scaled = np.clip(b_scaled, 0.0, 1.0)

    # Close loop
    N = len(feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    a_plot = np.concatenate([a_scaled, a_scaled[:1]])
    b_plot = np.concatenate([b_scaled, b_scaled[:1]])

    fig = plt.figure(figsize=(9, 6.5))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, a_plot, linewidth=2, label=a_label, color=color_a)
    ax.fill(angles, a_plot, alpha=0.10, color=color_a)

    ax.plot(angles, b_plot, linewidth=2, label=b_label, color=color_b)
    ax.fill(angles, b_plot, alpha=0.10, color=color_b)

    ax.set_xticks(angles[:-1])

    # Label each axis with feature name + original range
    xticklabels = []
    for f, (mn, mx) in zip(feats, ranges):
        xticklabels.append(f"{f}\n[{mn:g}, {mx:g}]")
    ax.set_xticklabels(xticklabels, fontsize=8)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)

    if title:
        ax.set_title(title + "\n(per-feature ranges from beforeValidation)", pad=20, fontsize=11)
    else:
        ax.set_title("Pair radar (per-feature ranges from beforeValidation)", pad=20, fontsize=11)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    ax.grid(True)

    # Annotate original values
    if annotate_values:
        for i, ang in enumerate(angles[:-1]):
            r = max(a_scaled[i], b_scaled[i]) + 0.07
            r = min(r, 1.08)
            ax.text(ang, r, f"A:{a[i]:g}\nB:{b[i]:g}", fontsize=7, ha="center", va="center")

    fig.tight_layout()

    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------
# Neo4j helpers
# ---------------------------

_SAFE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _validate_cypher_identifier(name: str, kind: str) -> str:
    if not isinstance(name, str) or not _SAFE_NAME_RE.match(name):
        raise ValueError(f"Unsafe {kind} identifier: {name!r}")
    return name

def fetch_node_infos_by_outid(db, label: str, outids, extra_props):
    label = _validate_cypher_identifier(label, "label")
    extra_props = list(extra_props or [])
    for p in extra_props:
        _validate_cypher_identifier(p, "property")

    outids = [int(x) for x in outids if not (x is None or (isinstance(x, float) and np.isnan(x)))]
    if not outids:
        return {}
    if not extra_props:
        return {oid: {} for oid in outids}

    proj = "n{ " + ", ".join(f".`{p}`" for p in extra_props) + " }"
    cypher = f"""
    MATCH (n:`{label}`)
    WHERE id(n) IN $ids
    RETURN id(n) AS outid, {proj} AS info
    """

    #if hasattr(db, "query"):
    #    records = db.query(cypher, {"ids": outids})
    #elif hasattr(db, "run"):
    #    records = db.run(cypher, {"ids": outids})
    #else:
    #    raise AttributeError("Neo4j connector 'db' must have a .query(...) or .run(...) method.")
    records = db.execute_query(cypher, {"ids": outids})

    info_map = {}
    for r in records:
        oid = int(r["outid"])
        info_map[oid] = dict(r.get("info") or {})

    for oid in outids:
        info_map.setdefault(int(oid), {})
    return info_map

def format_node_info(outid, info: dict):
    if not info:
        return f"outid={int(outid)}"
    parts = [f"{k}={info.get(k)}" for k in info.keys()]
    return f"outid={int(outid)} | " + " | ".join(parts)


# ---------------------------
# Stats helpers (original data)
# ---------------------------

def _cluster_feature_stats_original(beforeValidation, bv_map, outids, feat_idx, feat_names, top_n=3):
    """
    Compute mean/std for original values for given feature indices within a cluster.

    beforeValidation: array with outid in col 0, features in cols 1..
    bv_map: dict[outid_float] -> row index in beforeValidation
    outids: list[int] for rows in cluster (from "all_rows" aligned to data)
    feat_idx: indices in data/features space (0-based, no outid)
    feat_names: names aligned with feat_idx values
    """
    rows = []
    for oid in outids:
        # keys in bv_map come from CSV floats; use float(oid)
        r = bv_map.get(float(oid), None)
        if r is not None:
            rows.append(r)

    if not rows or len(feat_idx) == 0:
        return pd.DataFrame(columns=["feature", "mean", "std", "n_non_nan"]).set_index("feature")

    # values: (n_rows, n_feats) from original data
    # +1 because col0 is outid
    vals = beforeValidation[np.array(rows, dtype=int)[:, None], (np.array(feat_idx, dtype=int) + 1)[None, :]]
    vals = vals.astype(float)

    means = np.nanmean(vals, axis=0)

    # sample std (ddof=1) when possible; else nan
    stds = np.full(vals.shape[1], np.nan, dtype=float)
    n_non_nan = np.sum(~np.isnan(vals), axis=0)
    for j in range(vals.shape[1]):
        if n_non_nan[j] >= 2:
            stds[j] = np.nanstd(vals[:, j], ddof=1)
        elif n_non_nan[j] == 1:
            stds[j] = 0.0  # only one value -> no dispersion
        else:
            stds[j] = np.nan

    df = pd.DataFrame({
        "feature": feat_names,
        "mean": means,
        "std": stds,
        "n_non_nan": n_non_nan
    }).set_index("feature")

    # sort by std (nan last)
    df_sorted = df.sort_values(by=["std", "n_non_nan"], ascending=[True, False], na_position="last")
    return df_sorted.head(top_n)


# ---------------------------
# Main function
# ---------------------------

def top_k_pairs_print_original_side_by_side_with_neo4j_and_cluster_stats(
    data,
    sol,
    feature,
    all_rows,
    beforeValidation,
    db,
    node_label: str,
    extra_props=None,
    k=5,
    max_features=None,
    float_fmt="{:.4f}",
    show_diff_row=True,
    top_n_low_std_features=3,
    radar_dir=None,
    #radar_scale="pair_minmax",
    radar_use_percentiles=False,
    radar_p_low=1.0,
    radar_p_high=99.0,
):
    """
    Adds, per cluster:
      - stats on ORIGINAL values for clustering features (sol.sol == -1):
        mean/std, and prints top-N with lowest std.
      - then prints top-k diverse pairs (computed on preprocessed comparison features sol.sol == 1),
        showing ORIGINAL values side-by-side and Neo4j extra info.
    """
    X = np.asarray(data)
    membership = np.asarray(sol.membership)
    feat_partition = np.asarray(sol.sol).ravel()

    feature = np.asarray(feature)
    # Build global feature ranges from ORIGINAL values (beforeValidation)
    # This creates a per-feature min/max (or percentile) used for radar scaling.
    feature_ranges = build_feature_ranges_from_beforeValidation(
        beforeValidation=beforeValidation,
        feature_names=feature,
        use_percentiles=radar_use_percentiles,
        p_low=radar_p_low,
        p_high=radar_p_high,
    )

    all_rows = np.asarray(all_rows, dtype=float)
    beforeValidation = np.asarray(beforeValidation, dtype=float)

    # Indices of comparison features (for pair scoring & printing)
    comp_idx = np.where(feat_partition == 1)[0]
    if comp_idx.size == 0:
        raise ValueError("No comparison features (sol.sol == 1).")

    # Indices of clustering features (for per-cluster stats)
    clust_idx = np.where(feat_partition == -1)[0]
    if clust_idx.size == 0:
        # not fatal, but user asked; still proceed
        clust_names = np.array([], dtype=object)
    else:
        clust_names = feature[clust_idx]

    # Preprocessed data restricted to comparison features (used for scoring pairs)
    Xc = X[:, comp_idx]
    comp_names = feature[comp_idx]

    # Map outid -> row index in beforeValidation
    bv_outids = beforeValidation[:, 0]
    bv_map = {}
    for r, oid in enumerate(bv_outids):
        if np.isnan(oid):
            continue
        if oid not in bv_map:
            bv_map[oid] = r

    # Ensure alignment between data and all_rows
    if all_rows.shape[0] != X.shape[0]:
        raise ValueError(
            f"Row count mismatch: data has {X.shape[0]} rows but all_rows has {all_rows.shape[0]} rows. "
            "They must be aligned row-by-row."
        )
    data_outids = all_rows[:, 0]

    labels = np.unique(membership)

    for lab in labels:
        idx = np.where(membership == lab)[0]
        m = idx.size

        print("\n" + "=" * 130)
        print(f"CLUSTER {lab}  (size={m})")

        if m < 1:
            print("Empty cluster.")
            continue

        # Cluster outids (for stats + neo4j prefetch)
        cluster_outids = [int(x) for x in data_outids[idx] if not np.isnan(x)]

        # ---- NEW: per-cluster stats on ORIGINAL values for clustering features ----
        if clust_idx.size > 0:
            stats_df = _cluster_feature_stats_original(
                beforeValidation=beforeValidation,
                bv_map=bv_map,
                outids=cluster_outids,
                feat_idx=clust_idx,
                feat_names=clust_names,
                top_n=top_n_low_std_features
            )

            print(f"\nTop {top_n_low_std_features} clustering features (sol.sol == -1) with lowest std (ORIGINAL scale):")
            if stats_df.empty:
                print("  (no stats available: missing outids or no original rows found)")
            else:
                with pd.option_context(
                    "display.max_columns", None,
                    "display.width", 220,
                    "display.float_format", lambda x: float_fmt.format(x)
                ):
                    print(stats_df)
        else:
            print("\nNo clustering features found (sol.sol has no -1).")

        if m < 2:
            print("\nNot enough points for pairs.")
            continue

        # Pre-fetch Neo4j info for all nodes in this cluster (batch)
        neo_info = fetch_node_infos_by_outid(db, node_label, cluster_outids, extra_props or [])

        # ---- Pair selection based on preprocessed comparison features ----
        A = Xc[idx]  # (m, d)
        D = np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2)
        iu, ju = np.triu_indices(m, k=1)
        dist_vals = D[iu, ju]

        kk = min(k, dist_vals.size)
        top_pos = np.argpartition(dist_vals, -kk)[-kk:]
        top_pos = top_pos[np.argsort(dist_vals[top_pos])[::-1]]

        for rank, p in enumerate(top_pos, start=1):
            i = int(idx[iu[p]])
            j = int(idx[ju[p]])

            oid_i = data_outids[i]
            oid_j = data_outids[j]
            if np.isnan(oid_i) or np.isnan(oid_j):
                print(f"\nPair #{rank}: skipped (missing outid).")
                continue

            oid_i = int(oid_i)
            oid_j = int(oid_j)

            # Preprocessed score used for selecting pairs / ordering displayed features
            diff_p = np.abs(Xc[i] - Xc[j])
            score = float(diff_p.sum())

            # Original values for printing (fallback to all_rows if outid not found)
            ri = bv_map.get(float(oid_i), None)
            rj = bv_map.get(float(oid_j), None)

            if ri is not None:
                xi_orig = beforeValidation[ri, comp_idx + 1]
            else:
                xi_orig = all_rows[i, comp_idx + 1]

            if rj is not None:
                xj_orig = beforeValidation[rj, comp_idx + 1]
            else:
                xj_orig = all_rows[j, comp_idx + 1]

            # choose which comparison features to display: largest preprocessed diffs
            order = np.argsort(diff_p)[::-1]
            if max_features is not None:
                order = order[:max_features]

            cols = comp_names[order]
            xi_show = xi_orig[order]
            xj_show = xj_orig[order]

            info_i = neo_info.get(oid_i, {})
            info_j = neo_info.get(oid_j, {})
            rowA = format_node_info(oid_i, info_i)
            rowB = format_node_info(oid_j, info_j)

            print(f"\nPair #{rank}  score(preprocessed)={float_fmt.format(score)}")
            print(f"  A: {rowA}")
            print(f"  B: {rowB}")

            rows = {
                f"A ({rowA})": xi_show,
                f"B ({rowB})": xj_show,
            }
            if show_diff_row:
                rows["|diff| (original)"] = np.abs(xi_show - xj_show)

            df = pd.DataFrame(rows, index=cols).T

            with pd.option_context(
                "display.max_columns", None,
                "display.width", 240,
                "display.float_format", lambda x: float_fmt.format(x)
            ):
                print(df)
                # ---- Radar chart for this pair (A vs B) ----
                if radar_dir is not None:
                    fname = f"cluster_{lab}_pair_{rank}_A_{oid_i}_B_{oid_j}.png"
                    outpath = os.path.join(radar_dir, fname)

                    title = f"Cluster {lab} | Pair #{rank}\nA: {rowA}\nB: {rowB}"

                    plot_pair_radar(
                        feature_names=list(cols),
                        a_values=xi_show,
                        b_values=xj_show,
                        feature_ranges=feature_ranges,
                        a_label=f"A ({oid_i})",
                        b_label=f"B ({oid_j})",
                        title=title,
                        outpath=outpath,
                        annotate_values=True
                    )

            if max_features is not None and max_features < diff_p.size:
                hidden_orig_sum = float(np.abs(xi_orig - xj_orig).sum() - np.abs(xi_show - xj_show).sum())
                print(
                    f"... {diff_p.size - max_features} more comparison features not shown "
                    f"(hidden |diff| original sum = {float_fmt.format(hidden_orig_sum)})"
                )
