import re
import numpy as np
import pandas as pd

# ---------------------------
# Neo4j helpers
# ---------------------------

_SAFE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _validate_cypher_identifier(name: str, kind: str) -> str:
    """
    Validate label/property identifiers to safely embed into Cypher.
    Neo4j does not allow parameterizing labels/property keys directly,
    so we must embed them in the query string.
    """
    if not isinstance(name, str) or not _SAFE_NAME_RE.match(name):
        raise ValueError(f"Unsafe {kind} identifier: {name!r}")
    return name

def fetch_node_infos_by_outid(db, label: str, outids, extra_props):
    """
    Fetch extra properties for nodes with a given label and internal ids (outids).

    Returns:
        dict[outid] -> dict of {prop: value, ...}
    """
    label = _validate_cypher_identifier(label, "label")

    extra_props = list(extra_props or [])
    for p in extra_props:
        _validate_cypher_identifier(p, "property")

    # If no extra props requested, still return empty dicts for each outid
    outids = [int(x) for x in outids if not (x is None or (isinstance(x, float) and np.isnan(x)))]
    if not outids:
        return {}

    if not extra_props:
        return {oid: {} for oid in outids}

    # Build a map projection: n{.name,.foo,.bar}
    proj = "n{ " + ", ".join(f".`{p}`" for p in extra_props) + " }"

    cypher = f"""
    MATCH (n:`{label}`)
    WHERE id(n) IN $ids
    RETURN id(n) AS outid, {proj} AS info
    """

    # Your connector API may differ; support the common patterns.
    # Expect records iterable with keys 'outid' and 'info'.
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
        info = r.get("info") or {}
        info_map[oid] = dict(info)

    # Ensure every requested outid has an entry
    for oid in outids:
        info_map.setdefault(int(oid), {})

    return info_map

def format_node_info(outid, info: dict):
    """
    Compact string for row labels, e.g. outid=10 | name=Kubrick | born=1928
    """
    if not info:
        return f"outid={int(outid)}"
    parts = [f"{k}={info.get(k)}" for k in info.keys()]
    return f"outid={int(outid)} | " + " | ".join(parts)

# ---------------------------
# Main function
# ---------------------------

def top_k_pairs_print_original_side_by_side_with_neo4j(
    data,                 # preprocessed numeric matrix used for clustering (nodes)
    sol,                  # has sol.membership and sol.sol
    feature,              # feature names (len = n_features in data)
    all_rows,             # preprocessed rows WITH outid in col 0 (your "all")
    beforeValidation,     # original rows WITH outid in col 0
    db,                   # Neo4j connector
    node_label: str,      # e.g. "Director"
    extra_props=None,     # list[str], e.g. ["name", "country"]
    k=5,
    max_features=None,
    float_fmt="{:.4f}",
    show_diff_row=True
):
    """
    For each cluster, retrieve top-k pairs maximizing L1 distance over comparison features
    (sol.sol == 1), computed on preprocessed data, and print ORIGINAL values side-by-side,
    enriched with Neo4j node info fetched via outid/internal id.

    Assumptions (matching your loader):
      - all_rows[i][0] is outid for data row i
      - beforeValidation rows have same outid in column 0
      - feature names correspond to columns 1.. in beforeValidation/all_rows
        and to columns 0.. in data
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

        print("\n" + "=" * 120)
        print(f"CLUSTER {lab}  (size={m})")

        if m < 2:
            print("Not enough points for pairs.")
            continue

        # Pre-fetch Neo4j info for all nodes in this cluster (batch)
        cluster_outids = [int(x) for x in data_outids[idx] if not np.isnan(x)]
        neo_info = fetch_node_infos_by_outid(db, node_label, cluster_outids, extra_props or [])

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
            if np.isnan(oid_i) or np.isnan(oid_j):
                print(f"\nPair #{rank}: skipped (missing outid).")
                continue

            oid_i = int(oid_i)
            oid_j = int(oid_j)

            # Preprocessed score used for selecting pairs / ordering features
            diff_p = np.abs(Xc[i] - Xc[j])
            score = float(diff_p.sum())

            # Original values for printing (fallback to all_rows if outid not found)
            ri = bv_map.get(float(oid_i), None)  # keys stored as floats from CSV
            rj = bv_map.get(float(oid_j), None)

            if ri is not None:
                xi_orig = beforeValidation[ri, comp_idx + 1]
            else:
                xi_orig = all_rows[i, comp_idx + 1]

            if rj is not None:
                xj_orig = beforeValidation[rj, comp_idx + 1]
            else:
                xj_orig = all_rows[j, comp_idx + 1]

            # Select which features to display: largest preprocessed differences
            order = np.argsort(diff_p)[::-1]
            if max_features is not None:
                order = order[:max_features]

            cols = comp_names[order]
            xi_show = xi_orig[order]
            xj_show = xj_orig[order]

            # Neo4j info strings
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
                "display.width", 220,
                "display.float_format", lambda x: float_fmt.format(x)
            ):
                print(df)

            if max_features is not None and max_features < diff_p.size:
                hidden_orig_sum = float(np.abs(xi_orig - xj_orig).sum() - np.abs(xi_show - xj_show).sum())
                print(
                    f"... {diff_p.size - max_features} more comparison features not shown "
                    f"(hidden |diff| original sum = {float_fmt.format(hidden_orig_sum)})"
                )
