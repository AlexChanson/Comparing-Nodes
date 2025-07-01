from setuptools.command.dist_info import dist_info

from datasets import load_iris
from utility import *
from b_and_b import *

import numpy as np
from PrettyPrint import PrettyPrintTree


def solve_node(p_sol, dataset, k, method="fcm", max_iters=100, conv_criteria=10e-4, m = 2.0):
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
            if shift <= conv_criteria:
                membership = labels
                break
            centroids = new_centroids
        else:
            # Reached max_iters without convergence – warn the user.
            membership = labels
            print("Warning: k-means heuristic did not converge")
    elif method in ("fcm", "fuzzy"): #TODO should fuzzy parameter be thr same in both spaces ?
        rng = np.random.default_rng(0)

        # Initialize membership matrix U with shape (k, n_samples)
        U = rng.random((k, n_samples))
        U /= np.sum(U, axis=0, keepdims=True)

        # Initialize membership matrix in comparison space
        U_comp = rng.random((k, n_samples))
        U_comp /= np.sum(U, axis=0, keepdims=True)

        conv_check = False
        for iteration in range(max_iters):
            U_old = U.copy()
            U_comp_old = U_comp.copy()

            # Compute cluster centers
            U_m = U ** m
            U_comp_m = U_comp ** m
            centroids = (U_m @ X) / np.sum(U_m, axis=1, keepdims=True)
            centroids_comp = (U_comp_m @ X_comp) / np.sum(U_comp_m, axis=1, keepdims=True)

            # Compute squared distances (k, n_samples)
            dist_2 = np.sum((centroids[:, None, :] - X[None, :, :]) ** 2, axis=2)
            dist_2 = np.fmax(dist_2, 1e-12)  # avoid division by zero

            dist_comp_2 = np.sum( -np.abs(centroids_comp[:, None, :] - X_comp[None, :, :]) , axis=2)

            # Update U
            exponent = 1.0 / (m - 1)
            for j in range(k):
                ratio = dist_2[j:j + 1, :] / dist_2
                U[j, :] = 1.0 / np.sum(ratio ** exponent, axis=0)

                ratio_comp = dist_comp_2[j:j + 1, :] / dist_comp_2
                U_comp[j, :] = 1.0 / np.sum(ratio_comp ** exponent, axis=0)

            # Convergence check
            if np.max(np.abs(U - U_old)) <= conv_criteria and np.max(np.abs(U_comp - U_comp_old)) <= conv_criteria:
                conv_check = True
                break

        if not conv_check:
            print("Warning: convergence")

        # Hard labeling by maximum membership
        harmonic_mean = (2 * U_comp * U)/(U + U_comp)
        membership = np.argmax(harmonic_mean, axis=0)
    else:
        raise NotImplementedError

    return membership


def bnb(node):
    if not node.is_leaf():
        for idx, a in enumerate(node.mask()):
            if a == 0:
                for child in [node.branch(idx, "cluster"), node.branch(idx, "comparison")]:
                    if not child.is_feasible():
                        child.membership = None
                        child.obj = float("-inf")
                        if not child.is_leaf():# skip unfeasible leaf
                            bnb(child)
                    else:
                        child.membership = solve_node(child.mask(), data, k)
                        child.obj = eval_obj(child, data, child.membership)
                        bnb(child)


# Solution structure : vector of len |indicators| : 0 unused (default for partial solution / 1 used for comparison / - 1 used for clustering
if __name__ == '__main__':
    features, data = load_iris()

    k = 3
    root = Node().build_root(features)
    bnb(root)

    pt = PrettyPrintTree(lambda x: x.children, lambda x: str(x.sol).replace(" ", "") + ' ' + print_obj(x, data), orientation=PrettyPrintTree.Horizontal)
    pt(root)

    print(max_from_tree(root))

    test = Node()
    test.sol = [-1,-1,-1,-1]
    print(test.is_feasible())
    print(test.is_leaf())