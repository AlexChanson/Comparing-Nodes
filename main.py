from datasets import load_iris
import numpy as np


def solve_follower(attribution_mask, dataset, k, method="kmeans", max_iters=100):
    X = dataset[:, attribution_mask]  # mask attributes used for comparison or discarded
    n_samples, _ = X.shape

    cluster_partition = None

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
                break
            centroids = new_centroids
        else:
            # Reached max_iters without convergence – warn the user.
            print("Warning: k-means heuristic did not converge")
        return labels

    else:
        raise NotImplementedError

if __name__ == '__main__':
    features, data = load_iris()

    k = 3
    mask = [0,0,1,1]

    print(solve_follower(mask, data, k))