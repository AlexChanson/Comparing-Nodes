from datasets import *
from b_and_b import *
from clustering import *
from utility import *
from copy import copy
import numpy as np
from numpy.typing import NDArray
import heapq
from itertools import count
import argparse
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
import time
from sklearn.metrics import silhouette_score

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
        membership = fcmd_nico(X, X_comp, conv_criteria, k, m, max_iters)
    else:
        raise NotImplementedError

    return membership


def signature(l : list[int]):
    return "".join(map(str, l))

# you have to pass node_map = dict() when calling
#@jit()
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

#@jit()
def bnb_iterative(root: "Node", node_map: dict, max_depth=10, method="kmeans"):
    # Min-heap; we invert the priority for desired behavior.
    # Order is mostly irrelevant, but we still need a deterministic, stable order.
    # Priority schema:
    #   (-depth, tie)  => deeper nodes first (DFS-like) while still using a heap.
    heap = []
    tie = count()

    def push(node: "Node"):
        # base condition: do not push if node is leaf or depth limit reached
        if node.is_leaf() or node.depth == max_depth:
            return
        heapq.heappush(heap, ((-node.depth), next(tie), node))

    # Start from root (do not pre-compute membership/obj here, same as recursive entry)
    push(root)

    while heap:
        _, _, node = heapq.heappop(heap)

        # Base condition (mirrors the recursive guard at the top)
        if node.is_leaf() or node.depth == max_depth:
            continue

        # Expand by iterating mask and branching where a == 0
        for idx, a in enumerate(node.mask()):
            if a != 0:
                continue

            to_branch = []

            # cluster branch (set idx to -1)
            cls = copy(node.sol)
            cls[idx] = -1
            if signature(cls) not in node_map:
                to_branch.append(node.branch(idx, "cluster"))
            else:
                # Symmetric -> could link children if you maintain a children list
                # node.children.append(node_map[signature(cls)])
                pass

            # comparison branch (set idx to 1)
            cmp_ = copy(node.sol)
            cmp_[idx] = 1
            if signature(cmp_) not in node_map:
                to_branch.append(node.branch(idx, "comparison"))
            else:
                # node.children.append(node_map[signature(cmp_)])
                pass

            # Process children exactly like the recursive function
            for child in to_branch:
                if not child.is_feasible():
                    child.membership = None
                    child.obj = float("-inf")

                    if not child.is_leaf():
                        # Save non-leaf unfeasible nodes and expand them later
                        node_map[child.signature()] = child
                        push(child)  # equivalent to: bnb(child, ...)
                    else:
                        # Skip unfeasible leaf (do NOT store in map)
                        # node.prune_child(child) if you keep a children list
                        pass
                else:
                    # Feasible: solve + evaluate, save, and expand if not at base condition
                    child.membership = solve_node(child.mask(), data, k, method=method)
                    child.obj = child.eval_obj(data)
                    child.obj = child.eval_obj(data)

                    node_map[child.signature()] = child
                    push(child)  # recursion replaced by pushing if not base


def heur_exp(data, features, k, mtd, max_depth):
    root = Node().build_root(features)

    nodes: dict[Node] = {}
    bnb_iterative(root, nodes, method=mtd, max_depth=max_depth)

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

    W = construct_W.construct_W(data, neighbor_mode='knn', k=5)
    lap_scores = lap_score.lap_score(data, W=W)
    laplacians = np.asarray(list(lap_scores)).reshape(len(lap_scores), 1)
    m = kmeans(laplacians, 10e-4, 2, 100)
    c1_mean = lap_scores[m.astype(bool)].mean()
    c0_mean = lap_scores[~m.astype(bool)].mean()
    solution = np.ones_like(features)
    #c0 has lowest average laplacian
    if c1_mean > c0_mean:
        solution[~m.astype(bool)] = -1
    else:
        solution[m.astype(bool)] = -1

    h_sol = list(map(int,solution.tolist()))
    membership = solve_node(h_sol, data, k, method=mtd, max_iters=100)
    n = Node().from_starting(h_sol, membership,
                             si_obj(data, k, len(h_sol), derive_clustering_mask(h_sol), derive_comparison_mask(h_sol),
                                    membership))
    print("[Init] Smart-select finished with solution:", n)
    return n

def heur_random(data, features, k, mtd):

    h_sol = random_feasible(features)
    membership = solve_node(h_sol, data, k, method=mtd, max_iters=100)
    n = Node().from_starting(h_sol, membership,
                             si_obj(data, k, len(h_sol), derive_clustering_mask(h_sol), derive_comparison_mask(h_sol),
                                    membership))
    print("[Init] Random finished with solution:", n)
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
            if n.sol[i] != 0:
                dis = n.discard(i)
                if dis.is_feasible():
                    dis.membership = solve_node(dis.mask(), data, k, method=mtd, max_iters=100)
                    dis.obj = dis.eval_obj(data)
                    possible.append(dis)

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
    #print("[Heuristic] Local Search finished with solution:", n)
    return n

# Solution structure : vector of len |indicators| : 0 unused (default for partial solution / 1 used for comparison / - 1 used for clustering
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Please specify dataset name"
    )

    parser.add_argument("-ds", "--dataset", default="actors", help="Name of dataset (iris, airports, movies)")
    parser.add_argument("-k", "--k", default=3, help="Number of clusters")
    parser.add_argument("-s", "--steps", default=10, help="Local search max steps")
    parser.add_argument("-a", "--alpha", default=1.0, help="Alpha parameter")
    parser.add_argument("-m", "--method", default="sls", help="Method to use (ls : local search, exp : full tree enumeration, sls: 'smart' start local search)")
    parser.add_argument("-p", "--path", default="", help="Path to custom dataset")
    parser.add_argument("-d", "--delimiter", default=",", help="Delimiter for custom dataset")
    args = parser.parse_args()

    if args.dataset == "iris":
        features, data = load_iris()
    elif args.dataset == "airports":
        features, data = load_airports()
    elif args.dataset == "movies":
        features, data = load_movies()
    elif args.dataset == "directors":
        features, data = load_directors()
    elif args.dataset == "actors":
        features, data = load_actors()
    elif args.dataset == "custom":
        features, data = load_custom(args.path, args.delimiter)

    data = normalize(data)

    k = int(args.k)
    mtd = "fcm2"
    DISPLAY = False
    ls_steps = int(args.steps)

    st = time.process_time()
    st_w = time.time()

    if args.method == "ls":
        sols = []
        for start in range(5):
            sol_rd = heur_random(data, features, k, mtd=mtd)
            sols.append(heur_local_search(data, features, k, mtd=mtd, start=sol_rd, n_steps=ls_steps))
        sols = sorted(sols, key=lambda x: x.obj)
        print("[Heuristic] Local Search finished with solutions:", sols)
        print("[best solution]:", sols[-1])
        print("[Silhouette]", silhouette_score(data[:, sol_rd.derive_clustering_mask()], sol_rd.membership))

    elif args.method == "exp":
        sol_exp = heur_exp(data, features, k, mtd=mtd, max_depth=9)
        print("[best solution]:", sol_exp)
        print("[Silhouette]", silhouette_score(data[:, sol_exp.derive_clustering_mask()], sol_exp.membership))

    elif args.method == "sls":
        sol_patrick = heur_express(data, features, k, mtd=mtd)
        sol_patrick = heur_local_search(data, features, k, mtd=mtd, start=sol_patrick, n_steps=ls_steps)
        print("[best solution]:", sol_patrick)
        print("[Silhouette]", silhouette_score(data[:,sol_patrick.derive_clustering_mask()], sol_patrick.membership))

    elif args.method == "lp":
        sol_patrick = heur_express(data, features, k, mtd=mtd)
        print("[best solution]:", sol_patrick)
        print("[Silhouette]", silhouette_score(data[:, sol_patrick.derive_clustering_mask()], sol_patrick.membership))

    elif args.method == "rd":
        sol_rd = heur_random(data, features, k, mtd=mtd)
        print("[best solution]:", sol_rd)
        print("[Silhouette]", silhouette_score(data[:, sol_rd.derive_clustering_mask()], sol_rd.membership))

    et = time.process_time()
    et_w = time.time()

    # get execution time
    res = et - st
    res_w = et_w - st_w
    print('[CPU time]', res, 'seconds')
    print('[Wall time]', res_w, 'seconds')


