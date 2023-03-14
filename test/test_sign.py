"""
tests for the SIGN module and it's integration
"""

import unittest

import torch
from torch import tensor
import networkx as nx
import numpy as np
from argparse import Namespace
from torch_geometric.data import Data
from torch_geometric.utils.random import barabasi_albert_graph
import scipy.sparse as ssp

from models.gnn import SIGNEmbedding, SIGN
from test_params import OPT
from datasets.elph import HashedDynamicDataset


class SIGNTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        degree = 5
        p = 0.2
        self.x = torch.rand((self.n_nodes, 2))
        # self.G = nx.newman_watts_strogatz_graph(n=self.n_nodes, k=degree, p=p)
        # self.A = nx.adjacency_matrix(self.G)
        self.edge_index = barabasi_albert_graph(self.n_nodes, degree)
        self.edge_weight = torch.ones(self.edge_index.shape[1])
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.n_nodes, self.n_nodes))
        # self.edge_index = torch.tensor(np.array(self.G.edges)).T
        self.test_edges = tensor([[0, 1], [1, 2]])
        self.neg_test_edges = tensor([[0, 1], [2, 0]])

    def test_generate_sign_features(self):
        sign_k = 2
        n_features = 2
        opt = {'sample_size': None, 'dataset_name': 'Cora', 'model': 'hashing', 'hidden_dimension': 3, 'sign_k': sign_k}
        opt = {**OPT, **opt}
        args = Namespace(**opt)
        split = 'train'
        root = ('.')
        data = Data(torch.rand(self.n_nodes, n_features), self.edge_index, self.edge_weight)
        elph_dataset = HashedDynamicDataset(root, split, data, self.test_edges, self.neg_test_edges, args)
        x = elph_dataset.generate_sign_features(data, self.edge_index, self.edge_weight, sign_k)
        self.assertTrue(x.shape == (self.n_nodes, n_features * (sign_k + 1)))
        sign_k = 0
        x = elph_dataset.generate_sign_features(data, self.edge_index, self.edge_weight, sign_k)
        self.assertTrue(x.shape == data.x.shape)
