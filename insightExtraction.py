import numpy as np

def top_k_diverse_pairs_per_cluster(data, sol, k=5):
    """
    For each cluster label in sol.membership, return the top-k pairs (i, j)
    within that cluster that maximize sum(abs(x_i - x_j)) over comparison features
    (features where sol.sol == 1).

    Returns:
        dict[label] -> list of tuples (score, i, j)
        where i, j are indices in the original dataset.
    """
    X = np.asarray(data)
    membership = np.asarray(sol.membership)
    feat_partition = np.asarray(sol.sol).ravel()

    # Indices of features used for comparison (marked with 1)
    comp_idx = np.where(feat_partition == 1)[0]
    if comp_idx.size == 0:
        raise ValueError("No comparison features found: sol.sol has no entries equal to 1.")

    # Work only on comparison features
    Xc = X[:, comp_idx]

    results = {}
    labels = np.unique(membership)

    for lab in labels:
        idx = np.where(membership == lab)[0]
        m = idx.size
        if m < 2:
            results[lab] = []
            continue

        # Submatrix for this cluster
        A = Xc[idx]  # shape (m, d)

        # Pairwise L1 distances (sum of absolute differences) via broadcasting:
        # D[p, q] = sum_k |A[p,k] - A[q,k]|
        # Shape: (m, m)
        D = np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2)

        # Consider only upper triangle (p < q) to avoid duplicates and self-pairs
        iu, ju = np.triu_indices(m, k=1)
        dist_vals = D[iu, ju]

        # Get top-k indices among dist_vals (descending)
        kk = min(k, dist_vals.size)
        if kk == 0:
            results[lab] = []
            continue

        top_pos = np.argpartition(dist_vals, -kk)[-kk:]
        top_pos = top_pos[np.argsort(dist_vals[top_pos])[::-1]]  # sort descending

        # Convert back to original data indices
        top_pairs = []
        for p in top_pos:
            score = float(dist_vals[p])
            i = int(idx[iu[p]])
            j = int(idx[ju[p]])
            top_pairs.append((score, i, j))

        results[lab] = top_pairs

    return results


# ---- Example usage ----
# pairs_by_cluster = top_k_diverse_pairs_per_cluster(data, sol, k=5)
# for lab, pairs in pairs_by_cluster.items():
#     print(f"Cluster {lab}:")
#     for score, i, j in pairs:
#         print(f"  score={score:.3f} pair=({i}, {j})")
