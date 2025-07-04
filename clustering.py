import numpy as np
from numpy.typing import NDArray
from numba import njit, jit, objmode
from sklearn.metrics import pairwise_distances

@njit
def _random_unique_indices(n_samples: np.int64, k: np.int64):
    """
    Return `k` distinct random integers in the range [0, n_samples).
    Uses a rejection loop because Numba lacks np.random.choice(replace=False).
    """
    out = np.empty(k, dtype= np.int64)
    chosen = np.zeros(n_samples, dtype= np.int64)       # 0/1 flags
    filled = 0
    while filled < k:
        idx = np.random.randint(0, n_samples)
        if chosen[idx] == 0:
            chosen[idx] = 1
            out[filled] = idx
            filled += 1
    return out

@njit()
def kmeans(X, conv_criteria, k, max_iters):
    # --- Initialisation TODO kmeans++
    n_samples, n_features = X.shape

    # ---------- initialisation ----------------------------------------
    centroids = np.empty((k, n_features), dtype=np.float64)
    init_idx = _random_unique_indices(n_samples, k)
    for j in range(k):
        centroids[j, :] = X[init_idx[j], :]

    labels = np.empty(n_samples, dtype=np.int64)
    prev_centroids = centroids.copy()

    # ---------- main loop ---------------------------------------------
    for it in range(max_iters):

        # --- assignment step -----------------------------------------
        for i in range(n_samples):
            best_c = 0
            best_d = 0.0
            for j in range(k):
                d = 0.0
                for f in range(n_features):
                    diff = X[i, f] - centroids[j, f]
                    d += diff * diff
                if j == 0 or d < best_d:
                    best_d = d
                    best_c = j
            labels[i] = best_c

        # --- update step ---------------------------------------------
        sums = np.zeros((k, n_features), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)

        for i in range(n_samples):
            c = labels[i]
            counts[c] += 1
            for f in range(n_features):
                sums[c, f] += X[i, f]

        for j in range(k):
            if counts[j] == 0:                       # empty cluster
                rand_idx = np.random.randint(0, n_samples)
                centroids[j, :] = X[rand_idx, :]
            else:
                inv = 1.0 / counts[j]
                for f in range(n_features):
                    centroids[j, f] = sums[j, f] * inv

        # --- convergence check ---------------------------------------
        shift = 0.0
        for j in range(k):
            for f in range(n_features):
                diff = centroids[j, f] - prev_centroids[j, f]
                shift += diff * diff
        shift = np.sqrt(shift)

        if shift <= conv_criteria:
            return labels

        prev_centroids[:, :] = centroids[:, :]
    return labels

@jit(nopython=True)
def fcm_alex(X: NDArray[np.float64], X_comp: NDArray[np.float64], conv_criteria : float, k: int, m:float, max_iters:int):
    n_samples = X.shape[0]

    U = np.random.rand(k, n_samples)
    U_comp = np.random.rand(k, n_samples)

    U /= U.sum(axis=0)
    U_comp /= U_comp.sum(axis=0)

    exponent = 1.0 / (m - 1.0)

    conv_check = False
    for _ in range(max_iters):
        U_old = U.copy()
        U_comp_old = U_comp.copy()

        U_m = U ** m
        U_comp_m = U_comp ** m

        centroids = (U_m @ X) / U_m.sum(axis=1)[:, None]
        centroids_comp = (U_comp_m @ X_comp) / U_comp_m.sum(axis=1)[:, None]

        dist_2 = ((centroids[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        dist_2 = np.fmax(dist_2, 1e-12)  # avoid /0

        dist_comp_2 = -np.abs(centroids_comp[:, None, :] - X_comp[None, :, :]).sum(axis=2)

        for j in range(k):
            ratio = dist_2[j] / dist_2
            ratio_comp = dist_comp_2[j] / dist_comp_2
            U[j] = 1.0 / np.sum(ratio ** exponent, axis=0)
            U_comp[j] = 1.0 / np.sum(ratio_comp ** exponent, axis=0)

        if np.abs(U - U_old).max() <= conv_criteria and np.abs(U_comp - U_comp_old).max() <= conv_criteria:
            conv_check = True
            break

    arithmetic_mean = 0.5 * (U + U_comp)
    if not conv_check:
        print("Warning: convergence")
    return arithmetic_mean.argmax(axis=0)

@njit()
def fcm_nico(X: NDArray[np.float64], X_comp: NDArray[np.float64], conv_criteria : float, k: int, m:float, max_iters:int):
    n_samples = X.shape[0]

    U = np.random.rand(k, n_samples)
    U /= U.sum(axis=0)

    exponent = 1.0 / (m - 1.0)

    conv_check = False
    for _ in range(max_iters):
        U_old = U.copy()
        U_m = U ** m

        centroids = (U_m @ X) / U_m.sum(axis=1)[:, None]

        dist_2 = ((centroids[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        dist_2 = np.fmax(dist_2, 1e-12)  # avoid /0

        for j in range(k):
            ratio = dist_2[j] / dist_2
            U[j] = 1.0 / np.sum(ratio ** exponent, axis=0)

        if np.abs(U - U_old).max() <= conv_criteria:
            conv_check = True
            break

    if not conv_check:
        print("Warning: convergence")
    return np.argmax(U, axis=0)




