from copy import copy
from utility import *

class Node:
    def __init__(self):
        self.sol = None
        self.depth = 0
        self.parent = None
        self.children = []
        self.membership = None
        self.root = False
        self.obj = float('nan')

    def build_root(self, indicators):
        if len(indicators) < 2:
            raise ValueError("Must have at least 2 indicators (problem undefined)")
        self.sol = [0]*len(indicators)
        self.root = True
        return self

    def branch(self, indicator, assignment):
        if self.sol[indicator] != 0:
            raise ValueError("Indicator already assigned")
        n = self.__copy()
        n.depth += 1
        n.parent = self
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

    def is_leaf(self):
        for a in self.sol:
            if a == 0:
                return False
        return True

    def is_feasible(self):
        for a in self.sol:
            if a == 1:
                for b in self.sol:
                    if b != -1:
                        return True
        return False

def eval_obj(node, dataset, membership):
    if membership is None:
        return float("nan")
    k = max(membership)
    X_ = dataset[:, derive_clustering_mask(node.mask())]
    X = dataset[:, derive_comparison_mask(node.mask())]

    s = 0

    for c in range(k):
        indices = np.argwhere(membership == c) # get indices for cluster
        for i in indices:
            for j in indices:
                if i > j:
                    s += (1.0/len(indices)) * np.sum(np.abs(X[i] - X[j] ))
                    s -= (1.0/len(indices)) * np.sum((X_[i] - X_[j])**2)

    return s

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