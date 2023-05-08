"""
Testing the SEAL data structure
"""

import unittest
from argparse import Namespace

import torch
from torch import tensor
import scipy.sparse as ssp
from torch_geometric.data import Data

from src.datasets.seal import SEALDataset, SEALDynamicDataset, k_hop_subgraph, \
    construct_pyg_graph
from src.data import get_data
from test_params import OPT
from src.utils import get_src_dst_degree, get_pos_neg_edges


class SEALDatasetTests(unittest.TestCase):
    def setUp(self):
        self.edge_index = tensor([[0, 1, 2, 0, 2, 1, 2, 3], [1, 0, 0, 2, 1, 2, 3, 2]])
        self.edge_set = {(0, 1), (2, 0), (2, 1), (1, 2)}
        self.target_edges = [(0, 1), (1, 2), (0, 2), (1, 0)]
        self.edge_weight = torch.ones(self.edge_index.size(1), dtype=int)
        self.test_edges = tensor([[0, 1], [1, 2]])
        self.num_nodes = 4
        self.neg_test_edges = tensor([[0, 1], [2, 0]])
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.num_nodes, self.num_nodes))
        # create a graph with 2 isomorphic nodes 2 & 3
        self.iso_edge_index = tensor([[2, 2, 3, 3, 4, 0], [1, 4, 1, 4, 0, 1]])
        self.iso_edge_weight = torch.ones(self.iso_edge_index.size(0), dtype=int)
        self.iso_test_edges = tensor([[2, 3], [0, 0]])
        self.iso_num_nodes = 5
        self.args = OPT

    def test_k_hop_subgraph(self):
        num_hops = ratio_per_hop = 1
        directed = False
        max_nodes_per_hop = None
        if directed:
            A_csc = self.A.tocsc()
        else:
            A_csc = None
        x = torch.rand(self.num_nodes, 2)
        src, dst = 0, 1
        nodes, subgraph, dists, node_features, y = k_hop_subgraph(src, dst, num_hops, self.A, ratio_per_hop,
                                                                  max_nodes_per_hop, node_features=x, y=1,
                                                                  directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(nodes, subgraph, dists, node_features, y, node_label='drnl')
        self.assertTrue(data.edge_index.shape[1] == 4)  # 3 edges minus the 0-1 edge and undirected
        src, dst = 1, 2
        nodes, subgraph, dists, node_features, y = k_hop_subgraph(src, dst, num_hops, self.A, ratio_per_hop,
                                                                  max_nodes_per_hop, node_features=x, y=1,
                                                                  directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(nodes, subgraph, dists, node_features, y, node_label='drnl')

        self.assertTrue(data.edge_index.shape[1] == 6)  # 4 edges minus 1->2 and 2->1, but undirected
        src, dst = 2, 1
        nodes, subgraph, dists, node_features, y = k_hop_subgraph(src, dst, num_hops, self.A, ratio_per_hop,
                                                                  max_nodes_per_hop, node_features=x, y=1,
                                                                  directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(nodes, subgraph, dists, node_features, y, node_label='drnl')
        self.assertTrue(data.edge_index.shape[1] == 6)  # 4 edges minus 1->2 and 2->1

    def test_seal_dynamic_dataset(self):
        path = './dataset/seal_test_data'
        use_coalesce = False
        node_label = 'drnl'
        ratio_per_hop = 1.
        max_nodes_per_hop = None
        data = Data(torch.rand(self.num_nodes, 2), self.edge_index, torch.ones(self.edge_index.size(1), dtype=int))
        directed = False
        num_hops = 1
        train_dataset = SEALDynamicDataset(
            path, data, self.test_edges, self.neg_test_edges, num_hops=num_hops, percent=1.0, split='train',
            use_coalesce=use_coalesce,
            node_label=node_label,
            ratio_per_hop=ratio_per_hop,
            max_nodes_per_hop=max_nodes_per_hop,
            directed=directed,
        )
        labels = train_dataset.labels
        self.assertTrue(len(labels) == self.test_edges.size(1) + self.neg_test_edges.size(
            1))  # one prediction for each pos and neg edges
        self.assertTrue(sum(labels) == self.test_edges.size(1))  # pos edges are labelled 1 and neg edges labelled 0

    def test_seal_dataset(self):
        path = './dataset/seal_test_data'
        use_coalesce = False
        node_label = 'drnl'
        ratio_per_hop = 1.
        max_nodes_per_hop = None
        data = Data(torch.rand(self.num_nodes, 2), self.edge_index, torch.ones(self.edge_index.size(1), dtype=int))
        directed = False
        num_hops = 1
        train_dataset = SEALDataset(
            path, data, self.test_edges, self.neg_test_edges, num_hops=num_hops, percent=1.0, split='train',
            use_coalesce=use_coalesce,
            node_label=node_label,
            ratio_per_hop=ratio_per_hop,
            max_nodes_per_hop=max_nodes_per_hop,
            directed=directed,
        )
        labels = train_dataset.data.y
        self.assertTrue(labels.shape[0] == self.test_edges.size(1) + self.neg_test_edges.size(
            1))  # one prediction for each pos and neg edges
        self.assertTrue(sum(labels) == self.test_edges.size(1))  # pos edges are labelled 1 and neg edges labelled 0

    def test_get_data(self):
        """
        We use the pyg RandomLinkSplit object to create train / val / test splits. For link prediction edges play 2 roles
        1/ for message passing 2/ as supervision
        :return:
        """
        opt = {'sample_size': None, 'dataset_name': 'Cora', 'num_hops': 2, 'max_dist': 10, 'max_nodes_per_hop': 10,
               'data_appendix': None, 'val_pct': 0.1, 'test_pct': 0.2, 'train_sample': 1, 'dynamic_train': True,
               'dynamic_val': True, 'model': 'linear', 'dynamic_test': True, 'node_label': 'drnl', 'ratio_per_hop': 1}
        opt = {**OPT, **opt}
        args = Namespace(**opt)
        dataset, splits, directed, eval_metric = get_data(args)
        train, val, test = splits['train'], splits['valid'], splits['test']
        train_pos_edges, train_neg_edges = get_pos_neg_edges(train)
        # the default behaviour is 1 negative edge for each positive edge
        self.assertTrue(train_pos_edges.shape == train_neg_edges.shape)
        val_pos_edges, val_neg_edges = get_pos_neg_edges(val)
        self.assertTrue(val_pos_edges.shape == val_neg_edges.shape)
        test_pos_edges, test_neg_edges = get_pos_neg_edges(test)
        self.assertTrue(test_pos_edges.shape == test_neg_edges.shape)

    def test_get_src_dst_degree(self):
        temp1 = len(self.A[0].indices)
        self.assertTrue(temp1 == 2)
        src_deg, dst_deg = get_src_dst_degree(0, 1, self.A, 3)
        self.assertTrue(src_deg == 2)
        self.assertTrue(dst_deg == 2)
        src_deg, dst_deg = get_src_dst_degree(0, 1, self.A, 1)
        self.assertTrue(src_deg == 1)
        self.assertTrue(dst_deg == 1)
        src_deg, dst_deg = get_src_dst_degree(0, 1, self.A, None)
        self.assertTrue(src_deg == 2)
        self.assertTrue(dst_deg == 2)
        src_deg, dst_deg = get_src_dst_degree(0, 1, self.A, 0)
        self.assertTrue(src_deg == 0)
        self.assertTrue(dst_deg == 0)
