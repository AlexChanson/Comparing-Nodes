import numpy as np
from numba import njit
from scipy.stats import rankdata, describe
import pandas as pd
import random

try:
    from diptest import diptest as _diptest_fn
    _HAS_DIP = True
except ImportError:
    _HAS_DIP = False


# ---------- Core metrics ----------

def bimodality_coefficient(x: np.ndarray) -> float:
    """
    BC = (skew^2 + 1) / kurtosis  (Pearson moments; kurtosis is 3 for Gaussian)
    Rule of thumb: BC > ~0.55 suggests multimodality/bimodality.
    """
    x = x[np.isfinite(x)]
    n = x.shape[0]
    if n < 5:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s == 0:
        return 0.0
    z = (x - m) / s
    skew = np.mean(z**3)
    kurt = np.mean(z**4)  # Pearson kurtosis (not excess)
    return (skew**2 + 1.0) / kurt if kurt > 0 else np.nan

def cv_of_spacings(x: np.ndarray) -> float:
    """
    CV of consecutive gaps after sorting: sd(spacings)/mean(spacings).
    Smaller is more evenly spaced. Perfectly even => CV = 0.
    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan
    gaps = np.diff(x)
    mean_gap = np.mean(gaps)
    if mean_gap <= 0:
        # zero span (all equal) or degenerate → treat as not suitable for comparison
        return np.inf
    sd_gap = np.std(gaps, ddof=0)
    return sd_gap / mean_gap

# ---------- Categorization logic ----------

# used for the simple heuristics
def categorize_feature(
    x,
    bc_threshold: float = 0.55,   # > this ⇒ considered multimodal → "clustering"
    cv_max: float = 0.05,         # ≤ this ⇒ evenly spaced → "comparison"
    min_n: int = 5
):
    """
    Returns: dict with category, bc, cv_spacings, n, and brief reason.
    """
    x = x[np.isfinite(x)]
    n = x.shape[0]

    if n < min_n:
        return {
            "category": "unused",
            "n": n,
            "bc": np.nan,
            "cv_spacings": np.nan,
            "reason": f"Too few values (<{min_n})."
        }

    bc = bimodality_coefficient(x)
    cv = cv_of_spacings(x)


    if np.isfinite(cv) and cv <= cv_max:
        return {
            "category": "comparison",
            "n": n,
            "bc": float(bc),
            "cv_spacings": float(cv),
            "reason": f"Even spacing: CV={cv:.3f} ≤ {cv_max}."
        }

    if np.isfinite(bc) and bc > bc_threshold:
        return {
            "category": "clustering",
            "n": n,
            "bc": float(bc),
            "cv_spacings": float(cv),
            "reason": f"BC={bc:.3f} > threshold {bc_threshold} → likely ≥2 modes."
        }


    return {
        "category": "unused",
        "n": n,
        "bc": float(bc),
        "cv_spacings": float(cv),
        "reason": "Not bimodal and not evenly spaced."
    }


def analyze_features(
    x,
    maxClustFeat=2,
    minCompFeat=1,
    bc_threshold: float = 0.55,
    cv_max: float = 0.05,
    min_n: int = 5
):
    """
    features: dict like {"feat1": [..], "feat2": [..], ...}
    Returns: dict mapping feature name -> result dict from categorize_feature.
    """

    results=[]
    BC = []
    CV = []
    for i in range(x.shape[1]):
        BC.append(bimodality_coefficient(x[:i]))
        CV.append(cv_of_spacings(x[:i]))

    BC=np.asarray(BC)
    CV=np.asarray(CV)

    sortedByBC=np.argsort(BC)
    # reverse for clustering since large values preferred
    sortedByBC = sortedByBC[::-1]
    sortedByBC = sortedByBC[:maxClustFeat]

    sortedByCV=np.argsort(CV)[:minCompFeat]


    for i in range(x.shape[1]):
        n = x[:,i].shape[0]
        if n < min_n:
            return {
                "category": "unused",
                "n": n,
                "bc": np.nan,
                "cv_spacings": np.nan,
                "reason": f"Too few values (<{min_n})."
            }
        else:
            if i in sortedByCV:
                results.append({
                    "category": "comparison",
                    "n": len(x[:i]),
                    "bc": float(BC[i]),
                    "cv_spacings": float(CV[i]),
                    "reason": f" among minCompFeat top feature for comparison"
                })
            else:
                if i in sortedByBC:
                    results.append({
                        "category": "clustering",
                        "n": len(x[:i]),
                        "bc": float(BC[i]),
                        "cv_spacings": float(CV[i]),
                        "reason": f"top feature for clustering"
                    })
                else:
                    if CV[i] <= cv_max:
                        results.append({
                            "category": "comparison",
                            "n": len(x[:i]),
                            "bc": float(BC[i]),
                            "cv_spacings": float(CV[i]),
                            "reason": f"not top for clustering and score ok"
                        })
                    else:
                        results.append({
                            "category": "unused",
                            "n": len(x[:i]),
                            "bc": float(BC[i]),
                            "cv_spacings": float(CV[i]),
                            "reason": f"not top for clustering and score not ok"
                        } )

    return results



def analyze_features_OLD(
    x,
    bc_threshold: float = 0.55,
    cv_max: float = 0.05,
    min_n: int = 5
):
    """
    features: dict like {"feat1": [..], "feat2": [..], ...}
    Returns: dict mapping feature name -> result dict from categorize_feature.
    """
    results = []
    for i in range(x.shape[1]):
        results.append(  categorize_feature(
            x[:,i],
            bc_threshold=bc_threshold,
            cv_max=cv_max,
            min_n=min_n
        )
        )
    return results




def spacing_scores(x):
    """
    Compute multiple even-spacing scores for a numeric series x.

    Returns a dict with:
      - normalized_greenwood: [0,1], 0 = perfectly even
      - cv_spacings: coefficient of variation of gaps
      - r2_linear: R^2 of linear regression (index -> value)
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 2:
        raise ValueError("Series must have at least 2 numbers.")

    # --- Spacings ---
    spacings = np.diff(x)
    mean_spacing = np.mean(spacings)
    sd_spacing = np.std(spacings, ddof=0)

    # CV of spacings
    cv = sd_spacing / mean_spacing if mean_spacing > 0 else np.nan

    '''
    # --- Normalized Greenwood ---
    span = x[-1] - x[0]
    if span == 0:
        g_norm = 1.0
    else:
        u = (x - x[0]) / span
        u_full = np.concatenate(([0], u, [1]))
        d = np.diff(u_full)
        G = np.sum(d ** 2)
        G_min = 1.0 / (n + 1)  # perfectly even
        G_max = 1.0  # degenerate
        g_norm = (G - G_min) / (G_max - G_min)

    # --- R² from linear fit ---
    idx = np.arange(n)
    slope, intercept = np.polyfit(idx, x, 1)
    y_pred = intercept + slope * idx
    ss_res = np.sum((x - y_pred) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "normalized_greenwood": g_norm,
        "cv_spacings": cv,
        "r2_linear": r2
    }
    '''
    return cv





def quick_cluster_score(x):
    """
    Quick check if a 1D feature is a good candidate for clustering.

    Returns:
      - bimodality_coefficient: > ~0.55 suggests multimodality
      - dip: Hartigan's dip statistic (None if diptest not installed)
      - dip_pvalue: p-value (None if diptest not installed)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
        return {
            "n": n,
            "bimodality_coefficient": np.nan,
            "dip": None,
            "dip_pvalue": None,
            "notes": "Too few values (<5)."
        }

    # --- Bimodality Coefficient (BC) ---
    mean = np.mean(x)
    std = np.std(x, ddof=0)
    if std == 0:
        bc = 0.0
    else:
        z = (x - mean) / std
        skew = np.mean(z**3)
        kurt = np.mean(z**4)  # Pearson kurtosis (3 for Gaussian)
        bc = (skew**2 + 1) / kurt if kurt > 0 else np.nan

    # --- Dip test (optional) ---
    dip_val, dip_p = (None, None)
    if _HAS_DIP:
        try:
            dip_val, dip_p = _diptest_fn(np.sort(x))
        except Exception:
            pass

    return {
        "n": n,
        "bimodality_coefficient": float(bc),
        "dip": float(dip_val) if dip_val is not None else None,
        "dip_pvalue": float(dip_p) if dip_p is not None else None
    }

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


def pairwise_from_membership(membership):
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


def best_from_tree(nodes : dict):
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
    sol = [0]*len(features)
    while not feasible:
        for i in range(len(features)):
            sol[i] = random.choice([-1,0,1])
        feasible = is_feasible(sol)
    return sol


def bi_obj_check(root, data):
    sols = []
    x = [] #comparison obj
    y = []
    def internal(node):
        obj1, obj2 = node.eval_bi_obj(data)
        if node.is_feasible():
            sols.append(node.sol)
            x.append(obj1)
            y.append(obj2)
        if node.is_leaf():
            return 1
        else:
            return 1 + sum([internal(c) for c in node.children])
    sol_count =internal(root)
    return sols, x, y



def outer_join_features(df_left: pd.DataFrame,
                        df_right: pd.DataFrame,
                        id_left: str = "rootId",
                        id_right: str = "node_id",
                        out_id: str = "node_id") -> pd.DataFrame:
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
    overlap.discard(id_left); overlap.discard(id_right)
    for c in overlap:
        if f"{c}_r" in M.columns:
            M[c] = M[c].combine_first(M[f"{c}_r"])
            M.drop(columns=[f"{c}_r"], inplace=True)

    # Drop the original id columns, put unified id first, sort for readability
    M.drop(columns=[id_left, id_right], inplace=True, errors="ignore")
    cols = [out_id] + [c for c in M.columns if c != out_id]
    #print(cols)
    #print(M.columns)
    M = M[cols].sort_values(out_id).reset_index(drop=True)
    return M


import pandas as pd


def remove_rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows containing at least one null value from the DataFrame.
    Prints statistics about the number of rows before and after cleaning.

    Parameters:
        df (pd.DataFrame): Input dataframe

    Returns:
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
        "even_grid": [1, 2, 3, 4, 5, 6],                 # → comparison (evenly spaced)
        "bimodal":   [1,1.2,1.1, 5,5.1,5.2,5.3],         # → clustering (bimodal)
        "messy":     [1, 1, 3, 5, 5, 6.7, 6.8, 7.4],     # → unused (neither)
        "tiny":      [0, 1, 2],                          # too few points
    }

    out = analyze_features(feats)
    for k, v in out.items():
        print(k, "->", v["category"], "|", v["reason"], "| BC:", v["bc"], "CV:", v["cv_spacings"])