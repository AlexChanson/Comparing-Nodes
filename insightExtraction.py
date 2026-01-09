import numpy as np
import pandas as pd

def top_k_diverse_pairs_per_cluster(
    data,
    original_data,
    sol,
    feature,
    k=5,
    max_features=None,
    float_fmt="{:.4f}"
):
    """
    Compute top-k most different pairs per cluster using NORMALIZED data
    (sum of abs diffs over comparison features), but DISPLAY only ORIGINAL values
    and ORIGINAL absolute differences.

    Displays, for each selected feature:
      - original value for point i
      - original value for point j
      - |diff| in original space

    Parameters
    ----------
    data : (n_samples, n_features) normalized data (used for ranking)
    original_data : (n_samples, n_features) original data (used for display)
    sol : object with sol.membership and sol.sol
    feature : (n_features,) feature names
    k : top pairs per cluster
    max_features : max number of comparison features to display (largest normalized diffs)
    float_fmt : formatting for floats
    """
    X = np.asarray(data)
    X_orig = np.asarray(original_data)
    membership = np.asarray(sol.membership)
    feat_partition = np.asarray(sol.sol).ravel()
    feature = np.asarray(feature)

    if X.shape != X_orig.shape:
        raise ValueError("data and original_data must have the same shape.")

    comp_idx = np.where(feat_partition == 1)[0]
    if comp_idx.size == 0:
        raise ValueError("No comparison features (sol.sol == 1).")

    # Normalized comparison features for scoring
    Xc = X[:, comp_idx]
    # Original comparison features for display
    Xc_orig = X_orig[:, comp_idx]
    comp_names = feature[comp_idx]

    for lab in np.unique(membership):
        idx = np.where(membership == lab)[0]
        m = idx.size

        print("\n" + "=" * 80)
        print(f"CLUSTER {lab}  (size={m})")

        if m < 2:
            print("Not enough points for pairs.")
            continue

        A = Xc[idx]

        # Pairwise L1 distances on normalized data
        D = np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2)

        iu, ju = np.triu_indices(m, k=1)
        dist_vals = D[iu, ju]

        kk = min(k, dist_vals.size)
        top_pos = np.argpartition(dist_vals, -kk)[-kk:]
        top_pos = top_pos[np.argsort(dist_vals[top_pos])[::-1]]

        for rank, p in enumerate(top_pos, start=1):
            i = int(idx[iu[p]])
            j = int(idx[ju[p]])

            # Used only to rank/select features to display
            diff_norm = np.abs(Xc[i] - Xc[j])
            total_norm = diff_norm.sum()

            # What we display
            xi_orig = Xc_orig[i]
            xj_orig = Xc_orig[j]
            diff_orig = np.abs(xi_orig - xj_orig)

            print(f"\nPair #{rank}  (data[{i}] vs data[{j}])")
            print(f"Pair score (normalized, hidden per-feature) = {float_fmt.format(total_norm)}")

            # Show the most contributing features (by normalized diffs),
            # but display original values and original diffs.
            order = np.argsort(diff_norm)[::-1]
            if max_features is not None:
                order = order[:max_features]

            df = pd.DataFrame(
                {
                    f"data[{i}]": xi_orig[order],
                    f"data[{j}]": xj_orig[order],
                    "|diff|": diff_orig[order],
                },
                index=comp_names[order]
            )

            with pd.option_context("display.float_format", lambda x: float_fmt.format(x)):
                print(df)

            if max_features is not None and max_features < diff_norm.size:
                # Note: this "hidden" message is about omitted features, not normalized values shown.
                omitted = diff_norm.size - max_features
                print(f"... {omitted} more comparison features not shown.")


# Example call:
# top_k_diverse_pairs_per_cluster(
#     data=data,
#     original_data=original_data,
#     sol=sol,
#     feature=feature,
#     k=5,
#     max_features=8
# )
