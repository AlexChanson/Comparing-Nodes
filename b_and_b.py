from copy import copy
from utility import *
from numba import jit, njit
from numpy.typing import NDArray

class Node:
    def __init__(self):
        self.sol : list[int] = None
        self.depth : int = 0
        self.children : list[Node] = []
        self.membership : list[int] = None
        self.root : bool = False
        self.obj : float = float('nan')

    def build_root(self, indicators):
        if len(indicators) < 2:
            raise ValueError("Must have at least 2 indicators (problem undefined)")
        self.sol = [0]*len(indicators)
        self.root = True
        return self

    def branch(self, indicator:int, assignment:str):
        if self.sol[indicator] != 0:
            raise ValueError("Indicator already assigned")
        n = self.__copy()
        n.depth += 1
        self.children.append(n)
        if assignment.startswith("cl"):
            n.sol[indicator] = -1
        elif assignment.startswith("co"):
            n.sol[indicator] = 1
        else:
            raise ValueError("Unknown assignment : pick either clust or comp")
        return n

    def __copy(self):
        n = Node()
        n.sol = copy(self.sol)
        n.depth = self.depth
        return n

    def mask(self):
        return copy(self.sol)

    def signature(self):
        return "".join(map(str, self.sol))

    def derive_clustering_mask(self):
        return np.asarray(self.sol) == -1

    def prune_child(self, child):
        self.children.remove(child)

    def derive_comparison_mask(self):
        return np.asarray(self.sol) == 1

    def is_leaf(self):
        for a in self.sol:
            if a == 0:
                return False
        return True

    def is_feasible(self) -> bool:
        for a in self.sol:
            if a == 1:
                for b in self.sol:
                    if b == -1:
                        return True
        return False


def eval_obj(node : Node, dataset:NDArray[np.float64], membership : list[int]):
    if membership is None:
        return float("nan")
    k = max(membership)

    X_ = dataset[:, node.derive_clustering_mask()]
    clus_ratio = np.sum(node.derive_clustering_mask())/len(node.sol)

    X = dataset[:, node.derive_comparison_mask()]
    comp_ratio = np.sum(node.derive_comparison_mask()) / len(node.sol)

    s = 0
    for c in range(k):
        indices = np.argwhere(membership == c) # get indices for cluster
        n=len(indices)
        for i in indices:
            for j in indices:
                if i > j:
                    s += ( 2/(n*(n-1)) ) * comp_ratio * (1.0/len(indices)) * np.sum(np.abs(X[i] - X[j] ))
                    s -= (1 - clus_ratio) * (1.0/len(indices)) * np.sum((X_[i] - X_[j])**2)

    return s

def eval_bi_obj(node, dataset, membership):
    if membership is None:
        return float("nan"), float("nan")
    k = max(membership)

    X_ = dataset[:, derive_clustering_mask(node.mask())]
    clus_ratio = np.sum(derive_clustering_mask(node.mask()))/len(node.sol)

    X = dataset[:, derive_comparison_mask(node.mask())]
    comp_ratio = np.sum(derive_comparison_mask(node.mask())) / len(node.sol)

    s1 = 0
    s2 = 0
    for c in range(k):
        indices = np.argwhere(membership == c) # get indices for cluster
        n = len(indices)
        for i in indices:
            for j in indices:
                if i > j:
                    s1 += ( 2/(n*(n-1)) ) * comp_ratio * (1.0/len(indices)) * np.sum(np.abs(X[i] - X[j] ))
                    s2 += (1 - clus_ratio) * (1.0/len(indices)) * np.sum((X_[i] - X_[j])**2)

    return float(s1), float(s2)

def print_obj(node, data):
    if node.root:
        return "obj: infeasible (root)"
    elif not node.is_feasible():
        return "obj: infeasible"
    else:
        return "obj: " + str(round(eval_obj(node, data, node.membership),2))


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