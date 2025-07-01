from datasets import load_iris
from utility import *
from b_and_b import *

import numpy as np
from PrettyPrint import PrettyPrintTree


def solve_node(p_sol, dataset, k, method="kmeans", max_iters=100):
    X = dataset[:, derive_clustering_mask(p_sol)]  # mask attributes used for comparison or discarded
    X_comp = dataset[:, derive_comparison_mask(p_sol)]
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
                cluster_mask = labels == j
                if np.any(cluster_mask):
                    new_centroids[j] = X[cluster_mask].mean(axis=0)
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

def max_from_tree(node):
    if node.is_leaf():
        return eval_obj(node, data, node.membership), node.sol
    else:
        res = [max_from_tree(c) for c in node.children]
        res.append((eval_obj(node, data, node.membership), node.sol))
        v, s = res[0]
        for val, sol in res[1:]:
            if val > v:
                v = val
                s = sol
        return v, s

def bnb(node):
    if not node.is_leaf():
        for idx, a in enumerate(node.mask()):
            if a == 0:
                l = node.branch(idx, "cluster")
                l.membership = solve_node(l.mask(), data, k)
                r = node.branch(idx, "comparison")
                r.membership = solve_node(r.mask(), data, k)
                bnb(l)
                bnb(r)

# Solution structure : vector of len |indicators| : 0 unused (default for partial solution / 1 used for comparison / - 1 used for clustering
if __name__ == '__main__':
    features, data = load_iris()

    k = 3
    root = Node().build_root(features)
    bnb(root)

    pt = PrettyPrintTree(lambda x: x.children, lambda x: print_obj(x, data))
    pt(root)

    print(max_from_tree(root))