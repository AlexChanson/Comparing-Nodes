import numpy as np
from numba import njit
from scipy.stats import rankdata
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
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
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
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size

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

    if np.isfinite(bc) and bc > bc_threshold:
        return {
            "category": "clustering",
            "n": n,
            "bc": float(bc),
            "cv_spacings": float(cv),
            "reason": f"BC={bc:.3f} > threshold {bc_threshold} → likely ≥2 modes."
        }

    if np.isfinite(cv) and cv <= cv_max:
        return {
            "category": "comparison",
            "n": n,
            "bc": float(bc),
            "cv_spacings": float(cv),
            "reason": f"Even spacing: CV={cv:.3f} ≤ {cv_max}."
        }

    return {
        "category": "unused",
        "n": n,
        "bc": float(bc),
        "cv_spacings": float(cv),
        "reason": "Not bimodal and not evenly spaced."
    }

def analyze_features(
    features: dict,
    bc_threshold: float = 0.55,
    cv_max: float = 0.05,
    min_n: int = 5
):
    """
    features: dict like {"feat1": [..], "feat2": [..], ...}
    Returns: dict mapping feature name -> result dict from categorize_feature.
    """
    results = {}
    for name, series in features.items():
        results[name] = categorize_feature(
            series,
            bc_threshold=bc_threshold,
            cv_max=cv_max,
            min_n=min_n
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


def max_from_tree(node):
    if node.is_leaf():
        return node.obj, node.sol
    else:
        res = [max_from_tree(c) for c in node.children]
        res.append((node.obj, node.sol))
        v, s = res[0]
        for val, sol in res[1:]:
            if val > v:
                v = val
                s = sol
        return v, s


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


# ---------- Example ----------
if __name__ == "__main__":
    feats = {
        "even_grid": [1, 2, 3, 4, 5, 6],                 # → comparison (evenly spaced)
        "bimodal":   [1,1.2,1.1, 5,5.1,5.2,5.3],         # → clustering (bimodal)
        "messy":     [1, 1, 3, 5, 5, 6.7, 6.8, 7.4],     # → unused (neither)
        "tiny":      [0, 1, 2],                          # too few points
    }

    out = analyze_features(feats, bc_threshold=0.55, cv_max=0.05, min_n=5)
    for k, v in out.items():
        print(k, "->", v["category"], "|", v["reason"], "| BC:", v["bc"], "CV:", v["cv_spacings"])