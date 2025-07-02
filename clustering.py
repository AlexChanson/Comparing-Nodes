import numpy as np
from numpy.typing import NDArray
from numba import njit, jit, objmode
from sklearn.metrics import pairwise_distances


def kmeans(X, conv_criteria, k, max_iters, membership, n_samples):
    # --- Initialisation TODO kmeans++
    rng = np.random.default_rng(0)  # TODO random from sys time ...
    initial_idx = rng.choice(n_samples, size=k, replace=False)
    centroids = X[initial_idx]
    for iteration in range(max_iters):
        # --- Assignment step ---------------------------------------------
        # Distance from every sample to every centroid (squared Euclidean).
        # Shape: (n_samples, k)
        diff = X[:, None, :] - centroids[None, :, :]
        distances = np.sum(diff * diff, axis=2)
        labels = np.argmin(distances, axis=1)

        # --- Update step --------------------------------------------------
        new_centroids = np.empty_like(centroids)
        for j in range(k):
            cluster_mask = labels == j
            if np.any(cluster_mask):
                new_centroids[j] = X[cluster_mask].mean(axis=0)
            else:
                # Handle empty cluster by re‑initializing to a random point.
                new_centroids[j] = X[rng.integers(n_samples)]

        # --- Convergence check -------------------------------------------
        shift = np.linalg.norm(new_centroids - centroids)
        if shift <= conv_criteria:
            membership = labels
            break
        centroids = new_centroids
    else:
        # Reached max_iters without convergence – warn the user.
        membership = labels
        print("Warning: k-means heuristic did not converge")
    return membership

@jit(nopython=True)
def fcm_alex(X: NDArray[np.float64], X_comp: NDArray[np.float64], conv_criteria : float, k: int, m:float, max_iters:int, n_samples:int):
    n_samples = X.shape[0]

    U = np.random.rand(k, n_samples)
    U_comp = np.random.rand(k, n_samples)

    U /= U.sum(axis=0)
    U_comp /= U_comp.sum(axis=0)

    exponent = 1.0 / (m - 1.0)

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
            break

    arithmetic_mean = 0.5 * (U + U_comp)
    return arithmetic_mean.argmax(axis=0)


def fcm_nico(X, conv_criteria, k, m, max_iters, membership, n_samples):
    rng = np.random.default_rng(0)
    # Initialize membership matrix U with shape (k, n_samples)
    U = rng.random((k, n_samples))
    U /= np.sum(U, axis=0, keepdims=True)
    conv_check = False
    for iteration in range(max_iters):
        U_old = U.copy()

        # Compute cluster centers
        U_m = U ** m
        centroids = (U_m @ X) / np.sum(U_m, axis=1, keepdims=True)

        # Compute squared distances (k, n_samples)
        dist_2 = np.sum((centroids[:, None, :] - X[None, :, :]) ** 2, axis=2)
        dist_2 = np.fmax(dist_2, 1e-12)  # avoid division by zero

        # Update U
        exponent = 1.0 / (m - 1)
        for j in range(k):
            ratio = dist_2[j:j + 1, :] / dist_2
            U[j, :] = 1.0 / np.sum(ratio ** exponent, axis=0)

        # Convergence check
        if np.max(np.abs(U - U_old)) <= conv_criteria:
            conv_check = True
            break
    if not conv_check:
        print("Warning: convergence")
    # Hard labeling by maximum membership
    membership = np.argmax(U, axis=0)
    return membership




