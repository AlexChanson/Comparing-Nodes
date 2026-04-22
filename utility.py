import random

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata

def percentile_rank(data):
    """
    Compute the percentile‐rank of each element in `data`.
    Returns an array of floats in [0, 1].
    """
    arr = np.asarray(data, dtype=float)
    # rankdata: ranks from 1 to N, averaging ties
    ranks = rankdata(arr, method='average')
    # scale so that min rank → 0.0, max rank → 1.0
    return (ranks - 1) / (len(arr) - 1)


def pairwise_from_membership(membership) -> None:
    pairwise = np.zeros((len(membership), len(membership)))
    for i in range(len(membership)):
        for j in range(len(membership)):
            if membership[i] == membership[j]:
                pairwise[i, j] = 1


@njit()
def derive_clustering_mask(mask):
    return np.asarray(mask) == -1


@njit()
def derive_comparison_mask(mask):
    return np.asarray(mask) == 1


def best_from_tree(nodes: dict):
    best = None
    best_score = -np.inf
    for node in nodes.values():
        if node.obj > best_score:
            best = node
            best_score = node.obj
    return best


def is_feasible(sol) -> bool:
    for a in sol:
        if a == 1:
            for b in sol:
                if b == -1:
                    return True
    return False


def random_feasible(features):
    feasible = False
    sol = [0] * len(features)
    while not feasible:
        for i in range(len(features)):
            sol[i] = random.choice([-1, 0, 1])
        feasible = is_feasible(sol)
    return sol


def bi_obj_check(root, data):
    sols = []
    x = []  # comparison obj
    y = []

    def internal(node):
        obj1, obj2 = node.eval_bi_obj(data)
        if node.is_feasible():
            sols.append(node.sol)
            x.append(obj1)
            y.append(obj2)
        if node.is_leaf():
            return 1
        return 1 + sum([internal(c) for c in node.children])

    internal(root)
    return sols, x, y


def outer_join_features(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_left: str = "rootId",
    id_right: str = "node_id",
    out_id: str = "node_id",
) -> pd.DataFrame:
    # Work on copies; align dtypes to nullable Int64 so NaNs are allowed
    A = df_left.copy()
    B = df_right.copy()
    A[id_left] = pd.to_numeric(A[id_left], errors="raise").astype("Int64")
    B[id_right] = pd.to_numeric(B[id_right], errors="raise").astype("Int64")

    # Outer merge; if there are overlapping non-id columns, suffix the right-hand ones
    M = A.merge(B, how="outer", left_on=id_left, right_on=id_right, suffixes=("", "_r"))

    # Single unified id
    M[out_id] = M[id_left].combine_first(M[id_right]).astype("Int64")

    # If any feature columns exist on both sides with the same name, keep the left value
    # and fall back to the right-suffixed one where left is NaN.
    overlap = set(A.columns) & set(B.columns)
    overlap.discard(id_left)
    overlap.discard(id_right)
    for c in overlap:
        if f"{c}_r" in M.columns:
            M[c] = M[c].combine_first(M[f"{c}_r"])
            M = M.drop(columns=[f"{c}_r"])

    # Drop the original id columns, put unified id first, sort for readability
    M = M.drop(columns=[id_left, id_right], errors="ignore")
    cols = [out_id] + [c for c in M.columns if c != out_id]
    # print(cols)
    # print(M.columns)
    return M[cols].sort_values(out_id).reset_index(drop=True)


import pandas as pd


def remove_rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows containing at least one null value from the DataFrame.
    Prints statistics about the number of rows before and after cleaning.

    Parameters
    ----------
        df (pd.DataFrame): Input dataframe

    Returns
    -------
        pd.DataFrame: DataFrame without rows containing null values

    """
    initial_rows = len(df)

    # Drop rows with at least one null
    cleaned_df = df.dropna()
    remaining_rows = len(cleaned_df)
    excluded_rows = initial_rows - remaining_rows
    percentage_remaining = (remaining_rows / initial_rows * 100) if initial_rows > 0 else 0

    print(f"Initial rows: {initial_rows}")
    print(f"Excluded rows: {excluded_rows}")
    print(f"Remaining rows: {remaining_rows} ({percentage_remaining:.2f}%)")

    return cleaned_df


# ---------- Example ----------
if __name__ == "__main__":
    feats = {
        "even_grid": [1, 2, 3, 4, 5, 6],  # → comparison (evenly spaced)
        "bimodal": [1, 1.2, 1.1, 5, 5.1, 5.2, 5.3],  # → clustering (bimodal)
        "messy": [1, 1, 3, 5, 5, 6.7, 6.8, 7.4],  # → unused (neither)
        "tiny": [0, 1, 2],  # too few points
    }

    out = analyze_features(feats)
    for k, v in out.items():
        print(k, "->", v["category"], "|", v["reason"], "| BC:", v["bc"], "CV:", v["cv_spacings"])
