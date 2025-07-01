import numpy as np


def pairwise_from_membership(membership):
    pairwise = np.zeros((len(membership), len(membership)))
    for i in range(len(membership)):
        for j in range(len(membership)):
            if membership[i] == membership[j]:
                pairwise[i, j] = 1

def derive_clustering_mask(mask):
    return np.asarray(mask) == -1

def derive_comparison_mask(mask):
    return np.asarray(mask) == 1

