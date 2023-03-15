"""
tests for the BUDDY model
"""

import unittest
from argparse import Namespace

import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.data import Data
from torch_geometric.utils.random import barabasi_albert_graph
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import scipy.sparse as ssp
# from datasketch import MinHash, HyperLogLogPlusPlus
# import networkx as nx
from torch_scatter import scatter_min

from runners.train import train_gnn
from runners.inference import get_gnn_preds
from test_params import OPT, setup_seed
from models.elph import BUDDY, LinkPredictor
from utils import ROOT_DIR, select_embedding
from datasets.elph import HashedDynamicDataset
from hashing import ElphHashes

class ELPHTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        degree = 5
        self.num_features = 3
        self.x = torch.rand((self.n_nodes, self.num_features))
        edge_index = barabasi_albert_graph(self.n_nodes, degree)
        edge_index = to_undirected(edge_index)
        self.edge_index, _ = add_self_loops(edge_index)
        edge_weight = torch.ones(self.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix((edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.n_nodes, self.n_nodes))
        self.args = Namespace(**OPT)
        self.args.model = 'BUDDY'
        setup_seed(0)

    def test_propagate(self):
        num_features = self.x.shape[1]
        hash_hops = 2
        args = self.args
        args.max_hash_hops = hash_hops
        sgf = torch.rand((self.n_nodes, hash_hops * (hash_hops + 2)))
        gnn = BUDDY(args, num_features=num_features)
        x = gnn(sgf, self.x)

    def test_feature_forward(self):
        pass

    def test_embedding_forward(self):
        pass



    def test_forward(self):
        pass
