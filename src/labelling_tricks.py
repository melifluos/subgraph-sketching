"""
labelling tricks as described in
https://proceedings.neurips.cc/paper/2021/hash/4be49c79f233b4f4070794825c323733-Abstract.html
"""

import torch
import numpy as np
from scipy.sparse.csgraph import shortest_path


def drnl_hash_function(dist2src, dist2dst):
    """
    mapping from source and destination distances to a single node label e.g. (1,1)->2, (1,2)->3
    @param dist2src: Int Tensor[edges] shortest graph distance to source node
    @param dist2dst: Int Tensor[edges] shortest graph distance to source node
    @return: Int Tensor[edges] of labels
    """
    dist = dist2src + dist2dst

    dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    # the src and dst nodes always get a score of 1
    z[dist2src == 0] = 1
    z[dist2dst == 0] = 1
    return z


def get_drnl_lookup(max_dist, num_hops):
    """
    A lookup table from DRNL labels to index into a contiguous tensor. DRNL labels are not contiguous and this
    lookup table is used to index embedded labels
    """
    max_label = get_max_label('drnl', max_dist, num_hops)
    res_arr = [None] * (max_label + 1)
    res_arr[1] = (1, 0)
    for src in range(1, num_hops + 1):
        for dst in range(1, max_dist + 1):
            label = drnl_hash_function(torch.tensor([src]), torch.tensor([dst]))
            res_arr[label] = (src, dst)
    z_to_idx = {}
    idx_to_dst = {}
    counter = 0
    for idx, elem in enumerate(res_arr):
        if elem is not None:
            z_to_idx[idx] = counter
            idx_to_dst[counter] = (elem)
            counter += 1
    return z_to_idx, idx_to_dst


def get_max_label(method, max_dist, num_hops):
    if method in {'de', 'de+'}:
        max_label = max_dist
    elif method in {'drnl-', 'drnl'}:
        max_label = drnl_hash_function(torch.tensor([num_hops]), torch.tensor([max_dist])).item()
    else:
        raise NotImplementedError
    return max_label


def drnl_node_labeling(adj, src, dst, max_dist=100):
    """
    The heuristic proposed in "Link prediction based on graph neural networks". It is an integer value giving the 'distance'
    to the (src,dst) edge such that src = dst = 1, neighours of dst,src = 2 etc. It implements
    z = 1 + min(d_x, d_y) + (d//2)[d//2 + d%2 - 1] where d = d_x + d_y
    z is treated as a node label downstream. Even though the labels measures of distance from the central edge, they are treated as
    categorical objects and embedded in an embedding table of size max_z * hidden_dim
    @param adj:
    @param src:
    @param dst:
    @return:
    """
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)
    dist2src[dist2src > max_dist] = max_dist

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)
    dist2dst[dist2dst > max_dist] = max_dist

    z = drnl_hash_function(dist2src, dist2dst)
    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 1, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 1, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist

    return dist.to(torch.long)