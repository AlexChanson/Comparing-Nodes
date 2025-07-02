from datasets import load_iris
from b_and_b import *
from clustering import *
from utility import *

from copy import copy
import numpy as np
from numpy.typing import NDArray
from numba import njit
from PrettyPrint import PrettyPrintTree


@njit()
def solve_node(p_sol : list[int], dataset : NDArray[np.float64], k : int, method="kmeans", max_iters=100, conv_criteria=10e-4, m = 2.5):
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
        membership = kmeans(X, conv_criteria, k, max_iters, membership, n_samples)
    elif method in ("fcm", "fuzzy"): #TODO should fuzzy parameter be thr same in both spaces ?
        membership = fcm_alex(X, X_comp, conv_criteria, k, m, max_iters, n_samples)
    else:
        raise NotImplementedError

    return membership



# you have to pass node_map = dict() when calling
def bnb(node : Node, node_map : dict[Node], **params):
    def signature(l : list[int]):
        return "".join(map(str, l))

    if not node.is_leaf():
        for idx, a in enumerate(node.mask()):
            if a == 0:
                # check for symmetry
                to_branch = []

                cls = copy(node.sol)
                cls[idx] = -1
                if signature(cls) not in node_map:
                    to_branch.append(node.branch(idx, "cluster"))
                else:
                    pass
                    #node.children.append(node_map[signature(cls)])

                cmp = copy(node.sol)
                cmp[idx] = 1
                if signature(cmp) not in node_map:
                    to_branch.append(node.branch(idx, "comparison"))
                else:
                    pass
                    #node.children.append(node_map[signature(cmp)])

                # Branch
                for child in to_branch:
                    if not child.is_feasible():
                        child.membership = None
                        child.obj = float("-inf")
                        if not child.is_leaf():# skip unfeasible leaf
                            node_map[child.signature()] = child  #Save node in the hash map
                            bnb(child, node_map, **params)
                        else:
                            node.prune_child(child)
                    else:
                        child.membership = solve_node(child.mask(), data, k, **params)
                        child.obj = child.eval_obj(data)
                        node_map[child.signature()] = child #Save node in the hash map
                        bnb(child, node_map, **params)


# Solution structure : vector of len |indicators| : 0 unused (default for partial solution / 1 used for comparison / - 1 used for clustering
if __name__ == '__main__':
    features, data = load_iris()
    #TODO Min max normalisation (or robust scaling) of attributes to avoid messing the clustering metric and trivial solution

    k = 3
    mtd = "fcm"
    DISPLAY = False

    root = Node().build_root(features)

    nodes : dict[Node] = {}
    bnb(root, nodes, method=mtd)

    pt = PrettyPrintTree(lambda x: x.children, lambda x: str(x.sol).replace(" ", "") + ' ' + x.print_obj(data),
                         orientation=PrettyPrintTree.Horizontal)
    pt(root)

    print(max_from_tree(root))

    if DISPLAY:
        from matplotlib import pyplot as plt
        sols, x, y = bi_obj_check(root, data)
        plt.scatter(x, y)
        plt.xlabel("Comparison score")
        plt.ylabel("Clustering (variance, lower is better)")
        plt.title(f"Heuristic={mtd}, k={k}")
        plt.show()
