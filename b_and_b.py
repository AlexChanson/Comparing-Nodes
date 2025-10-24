from copy import copy

from numba import njit, types
from numba.types import Optional
from numpy.typing import NDArray
from utility import *

node_type = types.DeferredType()

spec = [
    ('sol', types.ListType(types.int64)),  # list[int]
    ('depth', types.int64),  # int
    ('parent', Optional(node_type)),  # Node | None
    ('membership', types.ListType(types.int64)),  # list[int]
    ('root', types.boolean),  # bool
    ('obj', types.float64),  # float
]


# @jitclass()
class Node:
    def __init__(self) -> None:
        self.sol: list[int] = []
        self.depth: int = 0
        self.parent: Node = None
        self.membership: list[int] = []
        self.root: bool = False
        self.obj: float = float('nan')

    def build_root(self, indicators):
        if len(indicators) < 2:
            msg = "Must have at least 2 indicators (problem undefined)"
            raise ValueError(msg)
        self.sol = [0] * len(indicators)
        self.root = True
        return self

    def from_starting(self, s, membership_matrix, objective):
        self.sol = s
        self.membership = membership_matrix
        self.obj = objective
        return self

    def branch(self, indicator: int, assignment: str):
        if self.sol[indicator] != 0:
            msg = "Indicator already assigned"
            raise ValueError(msg)
        n = self.__copy()
        n.depth += 1
        n.parent = self
        if assignment.startswith("cl"):
            n.sol[indicator] = -1
        elif assignment.startswith("co"):
            n.sol[indicator] = 1
        else:
            msg = "Unknown assignment : pick either clust or comp"
            raise ValueError(msg)
        return n

    def swap(self, indicator: int):
        n = self.__copy()
        n.depth += 1
        self.parent = self
        if self.sol[indicator] == 1:
            n.sol[indicator] = -1
            return n
        if self.sol[indicator] == -1:
            n.sol[indicator] = 1
            return n
        return None

    def discard(self, indicator: int):
        n = self.__copy()
        n.depth += 1
        self.parent = self
        n.sol[indicator] = 0
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

    def derive_comparison_mask(self):
        return np.asarray(self.sol) == 1

    def is_leaf(self) -> bool:
        return all(a != 0 for a in self.sol)

    def is_feasible(self) -> bool:
        for a in self.sol:
            if a == 1:
                for b in self.sol:
                    if b == -1:
                        return True
        return False

    def print_obj(self, data):
        if self.root:
            return "obj: infeasible (root)"
        if not self.is_feasible():
            return "obj: infeasible"
        return "obj: " + str(round(self.eval_obj(data), 2))

    def eval_obj(self, dataset: NDArray[np.float64]):
        if self.membership is None:
            return float("nan")
        k = max(self.membership)

        return si_obj(
            dataset, k, len(self.sol), self.derive_clustering_mask(), self.derive_comparison_mask(), self.membership
        )

    def eval_bi_obj(self, dataset):
        if self.membership is None:
            return float("nan"), float("nan")
        k = max(self.membership)

        s1, s2 = _bi_obj(
            dataset, k, len(self.sol), self.derive_clustering_mask(), self.derive_comparison_mask(), self.membership
        )

        return float(s1), float(s2)

    def __str__(self) -> str:
        return str(str(self.sol).replace(", ", ";") + " | " + str(self.obj))

    def __repr__(self) -> str:
        return str(str(self.sol).replace(", ", ";") + " | " + str(self.obj))


# node_type.define(Node.class_type.instance_type)


@njit
def _bi_obj(dataset, k, sol_len, cl_mask, co_mask, membership):
    X_ = dataset[:, cl_mask]
    clus_ratio = np.sum(cl_mask) / sol_len  # clustering dims / total dims
    X = dataset[:, co_mask]
    comp_ratio = np.sum(co_mask) / sol_len  # clustering dis / total dims
    s1 = 0
    s2 = 0
    for c in range(k):
        indices = np.argwhere(membership == c)  # get indices for cluster
        n = len(indices)
        for i in indices:
            for j in indices:
                if i > j:
                    s1 += comp_ratio * np.sum(np.abs(X[i] - X[j]))
                    s2 += (1 - clus_ratio) * np.sum((X_[i] - X_[j]) ** 2)
    return s1, s2


@njit
def si_obj(dataset, k, sol_len, cl_mask, co_mask, membership):
    X_ = dataset[:, cl_mask]
    clus_ratio = np.sum(cl_mask) / sol_len  # clustering dims / total dims
    X = dataset[:, co_mask]
    comp_ratio = np.sum(co_mask) / sol_len  # clustering dis / total dims
    s = 0
    for c in range(k):
        indices = np.argwhere(membership == c)  # get indices for cluster
        n = len(indices)
        for i in indices:
            for j in indices:
                if i > j:
                    # s += (2 / (n * (n - 1))) * comp_ratio * (1.0 / len(indices)) * np.sum(np.abs(X[i] - X[j]))
                    s += comp_ratio * np.sum(np.abs(X[i] - X[j]))
                    # s -= (1 - clus_ratio) * (1.0 / len(indices)) * np.sum((X_[i] - X_[j]) ** 2)
                    s -= (1 - clus_ratio) * np.sum((X_[i] - X_[j]) ** 2)

    return s
