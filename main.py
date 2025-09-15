from datasets import load_iris, normalize, load_airports
from b_and_b import *
from clustering import *
from utility import *

from multiprocessing import Pool
from copy import copy
import numpy as np
from numpy.typing import NDArray
from numba import njit
#from PrettyPrint import PrettyPrintTree


#@njit()
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
        membership = kmeans(X, conv_criteria, k, max_iters)
    elif method == "fcm": #TODO should fuzzy parameter be thr same in both spaces ?
        membership = fcm_alex(X, X_comp, conv_criteria, k, m, max_iters)
    elif method == "fcm2":
        membership = fcm_nico(X, X_comp, conv_criteria, k, m, max_iters)
    else:
        raise NotImplementedError

    return membership


def signature(l : list[int]):
    return "".join(map(str, l))

# you have to pass node_map = dict() when calling
def bnb(node : Node, node_map : dict[Node], max_depth, **params):
    if not (node.is_leaf() or node.depth == max_depth):
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
                            bnb(child, node_map, max_depth, **params)
                        else:
                            pass #node.prune_child(child)
                    else:
                        child.membership = solve_node(child.mask(), data, k, **params)
                        child.obj = child.eval_obj(data)
                        node_map[child.signature()] = child #Save node in the hash map
                        bnb(child, node_map, max_depth, **params)


def heur_exp(data, features, k, mtd, max_depth, **params):
    root = Node().build_root(features)

    nodes: dict[Node] = {}
    bnb(root, nodes, method=mtd, max_depth=max_depth, **params)

    #pt = PrettyPrintTree(lambda x: x.children, lambda x: str(x.sol).replace(" ", "") + ' ' + x.print_obj(data), orientation=PrettyPrintTree.Horizontal)
    #pt(root)

    sol = best_from_tree(nodes)

    #if DISPLAY:
    #    from matplotlib import pyplot as plt
    #    sols, x, y = bi_obj_check(root, data)
    #    plt.scatter(x, y)
    #    plt.xlabel("Comparison score")
    #    plt.ylabel("Clustering (variance, lower is better)")
    #    plt.title(f"Heuristic={mtd}, k={k}")
    #    plt.show()

    print("[Heuristic] Exponential finished with solution:", sol)
    return sol

def heur_express(data, features, k, mtd):
    h_assignement = analyze_features(data, maxClustFeat=min(5, len(features) - 1))

    h_sol = list(map(lambda x: x['category'], h_assignement))
    m = {'unused': 0, "clustering": -1, "comparison": 1}
    h_sol = list(map(lambda x: m[x], h_sol))
    membership = solve_node(h_sol, data, k, method=mtd, max_iters=100)
    n = Node().from_starting(h_sol, membership,
                             si_obj(data, k, len(h_sol), derive_clustering_mask(h_sol), derive_comparison_mask(h_sol),
                                    membership))
    print("[Heuristic] Express finished with solution:", n)
    return n

def heur_local_search(data, features, k, mtd, start, n_steps=5):
    n = start
    while n_steps > 0:
        print("[Local search] Step", n_steps)
        possible = []
        for i in range(len(features)):
            s = n.swap(i)
            if not s:
                l = n.branch(i, "cluster")
                r = n.branch(i, "comparison")
                l.membership = solve_node(l.mask(), data, k, method=mtd, max_iters=100)
                r.membership = solve_node(r.mask(), data, k, method=mtd, max_iters=100)
                l.obj = l.eval_obj(data)
                r.obj = r.eval_obj(data)
                possible.append(l)
                possible.append(r)
            elif s.is_feasible():
                s.membership = solve_node(s.mask(), data, k, method=mtd, max_iters=200)
                s.obj = s.eval_obj(data)
                possible.append(s)

        best_node = possible[0]
        best_obj = possible[0].obj

        for p in possible:
            if p.obj > best_obj:
                best_obj = p.obj
                best_node = p

        if best_obj < n.obj:
            print("Local minima found")
            break
        n = best_node
        n_steps -= 1
    print("[Heuristic] Local Search finished with solution:", n)
    return n

# Solution structure : vector of len |indicators| : 0 unused (default for partial solution / 1 used for comparison / - 1 used for clustering
if __name__ == '__main__':
    features, data = load_iris()
    data = normalize(data)

    k = 3
    mtd = "fcm2"
    DISPLAY = False

    sol_exp = heur_exp(data, features, k, mtd=mtd, max_depth=9)
    sol_patrick = heur_express(data, features, k, mtd=mtd)
    sol_ls = heur_local_search(data, features, k, mtd=mtd, start=sol_patrick)




