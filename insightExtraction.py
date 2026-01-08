import numpy as np
import pandas as pd

def top_k_diverse_pairs_per_cluster(
    data,
    sol,
    feature,               # ← NEW: feature names array
    k=5,
    max_features=None,
    float_fmt="{:.4f}"
):
    """
    For each cluster, find top-k pairs maximizing sum of absolute differences
    over comparison features (sol.sol == 1), and print analyst-friendly tables.

    Parameters
    ----------
    data : array-like (n_samples, n_features)
    sol  : object with attributes
           - sol.membership : cluster labels
           - sol.sol        : feature partition
    feature : array-like (n_features,)
        Feature names
    k : int
        Number of top pairs per cluster
    max_features : int or None
        Max number of comparison features to display per pair
    float_fmt : str
        Float formatting string
    """
    X = np.asarray(data)
    membership = np.asarray(sol.membership)
    feat_partition = np.asarray(sol.sol).ravel()
    feature = np.asarray(feature)

    # Indices of comparison features
    comp_idx = np.where(feat_partition == 1)[0]
    if comp_idx.size == 0:
        raise ValueError("No comparison features (sol.sol == 1).")

    Xc = X[:, comp_idx]
    comp_names = feature[comp_idx]

    labels = np.unique(membership)

    for lab in labels:
        idx = np.where(membership == lab)[0]
        m = idx.size

        print("\n" + "=" * 80)
        print(f"CLUSTER {lab}  (size={m})")

        if m < 2:
            print("Not enough points for pairs.")
            continue

        A = Xc[idx]

        # Pairwise L1 distances
        D = np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2)

        iu, ju = np.triu_indices(m, k=1)
        dist_vals = D[iu, ju]

        kk = min(k, dist_vals.size)
        top_pos = np.argpartition(dist_vals, -kk)[-kk:]
        top_pos = top_pos[np.argsort(dist_vals[top_pos])[::-1]]

        for rank, p in enumerate(top_pos, start=1):
            i = idx[iu[p]]
            j = idx[ju[p]]

            xi = Xc[i]
            xj = Xc[j]
            diff = np.abs(xi - xj)
            total = diff.sum()

            print(f"\nPair #{rank}  (data[{i}] vs data[{j}])")
            print(f"Total |diff| = {float_fmt.format(total)}")

            # Sort features by contribution
            order = np.argsort(diff)[::-1]
            if max_features is not None:
                order = order[:max_features]

            df = pd.DataFrame(
                {
                    "data[i]": xi[order],
                    "data[j]": xj[order],
                    "|diff|": diff[order],
                },
                index=comp_names[order]
            )

            with pd.option_context(
                "display.float_format",
                lambda x: float_fmt.format(x)
            ):
                print(df)

            if max_features is not None and max_features < diff.size:
                shown = diff[order].sum()
                hidden = total - shown
                print(
                    f"... {diff.size - max_features} more features "
                    f"(hidden |diff| sum = {float_fmt.format(hidden)})"
                )
