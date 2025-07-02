import numpy as np
from numba import njit

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
