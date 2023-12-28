"""
methods to rewire the data graph to improve computation including DropEdge
"""
from typing import Optional

# coding=utf-8
import numpy as np
import torch
import scipy.sparse as sp


# from utils import data_loader
# from normalization import fetch_normalization

# todo: make an enum of methods that the class can map
# todo: work out how to use *args and **kwargs to make this nice
class EdgeBuilder:
    def __init__(self, args):
        self.edge_dropout = args.edge_dropout
        self.remove_pos = args.remove_target_links
        self.add_negs = args.add_negative_links
        self.drop_mp_links = args.drop_message_passing_links

    def rewire_graph(self, links: torch.Tensor, labels: torch.Tensor, edge_index: torch.Tensor,
                     edge_weight: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rewire the graph according to the specified combination of methods
        @param pos: [batch_size, 2] tensor of positive edges
        @param negs: [batch_size * negs_per_pos, 2] tensor of negative edges
        @param edge_index: [2, n_edges] tensor of edges
        @param edge_weight: [n_edges] tensor of edge weights
        @return:
        """
        pos = links[labels == 1]
        negs = links[labels == 0]
        if self.remove_pos:
            edge_index, edge_weight = self.remove_target_links(pos, edge_index, edge_weight)
        if self.drop_mp_links:
            edge_index, edge_weight = self.drop_message_passing_links(edge_index, edge_weight)
        if self.add_negs:
            edge_index, edge_weight = self.add_negative_links(negs, edge_index, edge_weight)
        return edge_index, edge_weight

    def add_negative_links(self, negs, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        if self.edge_dropout != 0:
            indices = self.get_sample_indices(len(negs))
            negs = negs[indices]
        edge_index = torch.cat([edge_index, negs.T], dim=1)
        # todo: might not want to add with weight 1, but edge weight is currently not used
        if edge_weight is not None:
            edge_weight = torch.cat([edge_weight, torch.ones(negs.shape[0])])
        return edge_index, edge_weight

    def remove_target_links(self, targets, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        """
        Remove a set of target edges from the edge list. These would typically be the supervision training edges
        @param targets: [batch_size, 2] tensor of target edges
        @param edge_index: [2, n_edges]
        @param edge_weight: [n_edges]
        @return:
        """
        if self.edge_dropout != 0:
            indices = self.get_sample_indices(len(targets))
            targets = targets[indices]
        target_set = set([tuple(target) for target in targets.tolist()])
        mask = [tuple(link) not in target_set for link in edge_index.T.tolist()]
        mask = torch.tensor(mask)
        edge_index = edge_index[:, mask]
        if edge_weight is not None:
            edge_weight = edge_weight[mask]
        return edge_index, edge_weight

    def get_sample_indices(self, n):
        samples = int((1 - self.edge_dropout) * n)
        perm = np.random.permutation(range(n))
        indices = perm[:samples]
        return indices

    def drop_message_passing_links(self, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        rewires the input graph. May use various techniques such as
        1. removing batch supervision edges
        2. DropEdge
        3. Adding high probability local edges
        @return:
        """
        indices = self.get_sample_indices(edge_index.shape[1])
        edge_index = edge_index[:, indices]
        if edge_weight is not None:
            edge_weight = edge_weight[indices]
        return edge_index, edge_weight


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
