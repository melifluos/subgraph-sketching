import os
from distutils.util import strtobool

import numpy as np
import scipy
import torch
from scipy.stats import sem

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def neighbors(fringe, A, outgoing=True):
    """
    Retrieve neighbours of nodes within the fringe
    @param fringe: a set of nodeIDs
    @param A: scipy csr if outgoing = True, otherwise scipy csc
    @param outgoing: Boolean
    @return:
    """
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def get_src_dst_degree(src, dst, A, max_nodes):
    """
    currently this function assumes an undirected unweighted adjacency
    @param src: torch Tensor[Int] [edges]
    @param dst: torch Tensor[Int] [edges]
    @param A: scipy sparse CSR adjacency matrix
    @param max_nodes: caps the max degree
    @return:
    """
    src_degree = A[src].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    dst_degree = A[dst].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    return src_degree, dst_degree
