import numpy as np
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
