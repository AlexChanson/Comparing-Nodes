from datasets import load_iris
from utility import *

import numpy as np


def eval_leader_obj(mask, dataset, membership):
    k = max(membership)
    X = dataset[:, mask]

    s = 0

    for c in range(k):
        indices = np.argwhere(membership == c) # get indices for cluster
        for i in indices:
            for j in indices:
                if i > j:
                    s += np.sum(np.abs(X[i] - X[j] ))

    return s


def solve_follower(attribution_mask, dataset, k, method="kmeans", max_iters=100):
    X = dataset[:, attribution_mask]  # mask attributes used for comparison or discarded
    n_samples, _ = X.shape

    membership = None

    #idiot proofing
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if X.ndim != 2:
        raise ValueError("X must be a 2‑D array (n_samples, n_features)")
    if k > n_samples:
        raise ValueError("k cannot exceed the number of samples")

    if method == "kmeans":
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
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = X[mask].mean(axis=0)
                else:
                    # Handle empty cluster by re‑initializing to a random point.
                    new_centroids[j] = X[rng.integers(n_samples)]

            # --- Convergence check -------------------------------------------
            shift = np.linalg.norm(new_centroids - centroids)
            if shift <= 10e-4:
                membership = labels
                break
            centroids = new_centroids
        else:
            # Reached max_iters without convergence – warn the user.
            membership = labels
            print("Warning: k-means heuristic did not converge")
        return membership

    else:
        raise NotImplementedError



if __name__ == '__main__':
    features, data = load_iris()

    k = 3
    mask = np.asarray([False,False,True,True])
    mask2 = [True,True,False,False]

    membership = solve_follower(mask, data, k)
    print(membership)
    lobj = eval_leader_obj(mask2, data, membership)
    print(lobj)

    print(derive_comparison_mask(np.asarray([0,1,-1,1])))
    print(derive_clustering_mask(np.asarray([0,1,-1,1])))