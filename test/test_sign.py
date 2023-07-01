"""
tests for the SIGN module and it's integration
"""

import unittest

import torch
from torch import tensor
from argparse import Namespace
from torch_geometric.data import Data
from torch_geometric.utils.random import barabasi_albert_graph
import scipy.sparse as ssp

from src.models.gnn import SIGNEmbedding, SIGN
from test_params import OPT
from src.datasets.elph import HashDataset


class SIGNTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        degree = 5
        self.x = torch.rand((self.n_nodes, 2))
        self.edge_index = barabasi_albert_graph(self.n_nodes, degree)
        self.edge_weight = torch.ones(self.edge_index.shape[1])
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.n_nodes, self.n_nodes))
        self.test_edges = tensor([[0, 1], [1, 2]])
        self.neg_test_edges = tensor([[0, 1], [2, 0]])

    def test_generate_sign_features(self):
        sign_k = 2
        n_features = 2
        opt = {'sample_size': None, 'dataset_name': 'Cora', 'hidden_dimension': 3, 'sign_k': sign_k}
        opt = {**OPT, **opt}
        args = Namespace(**opt)
        split = 'train'
        root = ('.')
        data = Data(torch.rand(self.n_nodes, n_features), self.edge_index, self.edge_weight)
        elph_dataset = HashDataset(root, split, data, self.test_edges, self.neg_test_edges, args)
        x = elph_dataset._generate_sign_features(data, self.edge_index, self.edge_weight, sign_k)
        self.assertTrue(x.shape == (self.n_nodes, n_features * (sign_k + 1)))
        sign_k = 0
        x = elph_dataset._generate_sign_features(data, self.edge_index, self.edge_weight, sign_k)
        self.assertTrue(x.shape == data.x.shape)

    def test_sign_forward(self):
        hidden_channels = 3
        K = 2
        batch_size = 16
        # sign takes a batch of edges
        x = torch.rand((batch_size, 2, hidden_channels * (K + 1)))  # the original features + K propagated features
        xs = torch.split(x, K + 1, dim=-1)
        self.assertTrue(len(xs) == K + 1)
        self.assertTrue(xs[0].shape == xs[-1].shape)
        self.assertTrue(x.shape[0] == xs[0].shape[0])
        torch.manual_seed(0)
        sign = SIGN(hidden_channels, hidden_channels, hidden_channels, K, 0.5)
        out = sign(x)
        self.assertTrue(out.shape == (batch_size, 2, hidden_channels))

    def test_convs(self):
        hidden_channels = 5
        x = torch.rand((self.n_nodes, hidden_channels))
        torch.manual_seed(0)
        sign = SIGNEmbedding(hidden_channels, hidden_channels, hidden_channels, 2, 0.5)
        torch.manual_seed(0)
        sign_forward = sign(x, self.edge_index, self.n_nodes)
        self.assertTrue(sign_forward.shape == (self.n_nodes, hidden_channels))
