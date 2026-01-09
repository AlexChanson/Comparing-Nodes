import numpy as np
import pandas as pd

def top_k_diverse_pairs_per_cluster_original_values_side_by_side(
    data,                 # preprocessed numeric matrix used for clustering (nodes)
    sol,                  # clustering solution object: sol.membership, sol.sol
    feature,              # feature names (len = n_features in data)
    all_rows,             # preprocessed rows WITH outid in col 0 (your "all")
    beforeValidation,     # original rows WITH outid in col 0
    k=5,
    max_features=None,
    outid_name="outid",
    float_fmt="{:.4f}",
    show_diff_row=True
):
    """
    Finds, for each cluster, the top-k pairs maximizing L1 distance over comparison
    features (sol.sol == 1), computed on *preprocessed* data, but prints *original*
    (beforeValidation) values in a side-by-side layout: columns=features, rows=members.

    Assumptions (matching your loader):
      - all_rows[i][0] is outid for data row i
      - beforeValidation rows have same outid in column 0
      - feature names correspond to columns 1.. in all_rows/beforeValidation
        and to columns 0.. in data (nodes)
    """
    X = np.asarray(data)
    membership = np.asarray(sol.membership)
    feat_partition = np.asarray(sol.sol).ravel()

    feature = np.asarray(feature)
    all_rows = np.asarray(all_rows, dtype=float)
    beforeValidation = np.asarray(beforeValidation, dtype=float)

    # Comparison features indices (in X / feature array)
    comp_idx = np.where(feat_partition == 1)[0]
    if comp_idx.size == 0:
        raise ValueError("No comparison features (sol.sol == 1).")

    # Preprocessed data restricted to comparison features (used for scoring pairs)
    Xc = X[:, comp_idx]
    comp_names = feature[comp_idx]

    # ---- Build mapping: outid -> row index in beforeValidation ----
    bv_outids = beforeValidation[:, 0]
    bv_map = {}
    for r, oid in enumerate(bv_outids):
        if np.isnan(oid):
            continue
        if oid not in bv_map:
            bv_map[oid] = r

    # ---- Map each data row to an outid via all_rows ----
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

        print("\n" + "=" * 110)
        print(f"CLUSTER {lab}  (size={m})")

        if m < 2:
            print("Not enough points for pairs.")
            continue

        A = Xc[idx]  # (m, d)

        # Pairwise L1 distances on preprocessed comparison features
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

            ri = bv_map.get(oid_i, None)
            rj = bv_map.get(oid_j, None)

            # preprocessed diffs (used to rank/display top features)
            diff_p = np.abs(Xc[i] - Xc[j])
            score = float(diff_p.sum())

            print(f"\nPair #{rank}  score(preprocessed)={float_fmt.format(score)}")
            print(f"  A: data[{i}]  {outid_name}={oid_i}")
            print(f"  B: data[{j}]  {outid_name}={oid_j}")

            # original values (fallback to all_rows if missing)
            if ri is not None:
                xi_orig = beforeValidation[ri, comp_idx + 1]
            else:
                xi_orig = all_rows[i, comp_idx + 1]

            if rj is not None:
                xj_orig = beforeValidation[rj, comp_idx + 1]
            else:
                xj_orig = all_rows[j, comp_idx + 1]

            # choose which features to display: largest preprocessed differences
            order = np.argsort(diff_p)[::-1]
            if max_features is not None:
                order = order[:max_features]

            cols = comp_names[order]
            xi_show = xi_orig[order]
            xj_show = xj_orig[order]

            # 2-row (or 3-row) side-by-side table with features as columns
            rows = {
                f"A ({outid_name}={oid_i})": xi_show,
                f"B ({outid_name}={oid_j})": xj_show,
            }
            if show_diff_row:
                rows["|diff| (original)"] = np.abs(xi_show - xj_show)

            df = pd.DataFrame(rows, index=cols).T

            with pd.option_context(
                "display.max_columns", None,
                "display.width", 200,
                "display.float_format", lambda x: float_fmt.format(x)
            ):
                print(df)

            # Optional: how much difference is hidden (on original scale)
            if max_features is not None and max_features < diff_p.size:
                hidden_orig_sum = float(np.abs(xi_orig - xj_orig).sum() - np.abs(xi_show - xj_show).sum())
                print(
                    f"... {diff_p.size - max_features} more comparison features not shown "
                    f"(hidden |diff| original sum = {float_fmt.format(hidden_orig_sum)})"
                )
