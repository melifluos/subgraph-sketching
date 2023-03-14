"""
Testing the subgraph sketching dataset objects
"""
import unittest
from argparse import Namespace
import os

import torch
from torch import tensor
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.utils.random import barabasi_albert_graph
import scipy.sparse as ssp

from datasets.elph import HashedDynamicDataset, make_train_eval_data
from test_params import OPT
from utils import ROOT_DIR
from hashing import ElphHashes


class ELPHDatasetTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        degree = 5
        p = 0.2
        self.x = torch.rand((self.n_nodes, 2))
        self.edge_index = barabasi_albert_graph(self.n_nodes, degree)
        self.edge_weight = torch.ones(self.edge_index.shape[1])
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.n_nodes, self.n_nodes))
        self.test_edges = tensor([[0, 1], [1, 2]])
        self.neg_test_edges = tensor([[0, 1], [2, 0]])
        self.args = Namespace(**OPT)

    def test_HashedDynamicDataset(self):
        torch.manual_seed(0)
        pos_edges = torch.randint(self.n_nodes, (10, 2))
        neg_edges = torch.randint(self.n_nodes, (10, 2))
        split = 'test'
        max_hops = 2
        # ei = torch.tensor(list(self.G.edges())).T
        ei = self.edge_index
        data = Data(self.x, ei)
        root = f'{ROOT_DIR}/test/test_HashedDynamicDataset'
        hash_name = f'{root}_{split}_hashcache.pt'
        cards_name = f'{root}_{split}_cardcache.pt'
        eh = ElphHashes(self.args)
        if os.path.exists(hash_name) and os.path.exists(cards_name):
            hashes = torch.load(hash_name)
            cards = torch.load(cards_name)
        else:
            hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
            torch.save(hashes, hash_name)
            torch.save(cards, cards_name)
        all_edges = torch.cat([pos_edges, neg_edges], 0)
        structure_features = eh.get_subgraph_features(all_edges, hashes, cards)
        hdd = HashedDynamicDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                                   directed=False,
                                   load_features=True, load_hashes=True, use_zero_one=True,
                                   cache_structure_features=True)
        dl = DataLoader(hdd, batch_size=1,
                        shuffle=False, num_workers=1)
        for sf, elem in zip(structure_features, dl):
            sf_test = elem[0]
            self.assertTrue(torch.all(torch.eq(sf, sf_test)))

    def test_make_train_eval_dataset(self):
        self.args.max_hash_hops = 2
        torch.manual_seed(0)
        pos_edges = torch.randint(self.n_nodes, (10, 2))
        neg_edges = torch.randint(self.n_nodes, (10, 2))
        n_pos_samples = 8
        negs_per_pos = 10
        split = 'train'
        # ei = torch.tensor(list(self.G.edges())).T
        ei = self.edge_index
        data = Data(self.x, ei)
        root = f'{ROOT_DIR}/test/test_HashedDynamicDataset'
        hdd = HashedDynamicDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                                   directed=False,
                                   load_features=True, load_hashes=True, use_zero_one=True,
                                   cache_structure_features=True)
        train_eval_dataset = make_train_eval_data(self.args, hdd, self.n_nodes, n_pos_samples=n_pos_samples,
                                                  negs_per_pos=negs_per_pos)
        self.assertTrue(len(train_eval_dataset.links) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.labels) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.structure_features) == (negs_per_pos + 1) * n_pos_samples)
        self.args.use_RA = True
        hdd = HashedDynamicDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                                   directed=False,
                                   load_features=True, load_hashes=True, use_zero_one=True,
                                   cache_structure_features=True)
        train_eval_dataset = make_train_eval_data(self.args, hdd, self.n_nodes, n_pos_samples=n_pos_samples,
                                                  negs_per_pos=negs_per_pos)
        self.assertTrue(len(train_eval_dataset.RA) == (negs_per_pos + 1) * n_pos_samples)
