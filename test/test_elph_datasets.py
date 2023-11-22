"""
Testing the subgraph sketching dataset objects
"""
import unittest
from argparse import Namespace
import os

import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.utils.random import barabasi_albert_graph
import scipy.sparse as ssp

from src.datasets.elph import HashDataset, make_train_eval_data
from test_params import OPT
from src.utils import ROOT_DIR, get_same_source_negs
from src.hashing import ElphHashes


class ELPHDatasetTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        degree = 5
        self.n_edges = 10  # number of positive training edges
        self.x = torch.rand((self.n_nodes, 2))
        self.edge_index = barabasi_albert_graph(self.n_nodes, degree)
        self.edge_weight = torch.ones(self.edge_index.shape[1])
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.n_nodes, self.n_nodes))
        self.pos_edges = torch.randint(self.n_nodes, (self.n_edges, 2))
        self.neg_edges = torch.randint(self.n_nodes, (self.n_edges, 2))

        self.args = Namespace(**OPT)

    def test_HashedDynamicDataset(self):
        torch.manual_seed(0)
        self.args.model = 'BUDDY'
        split = 'test'
        ei = self.edge_index
        data = Data(self.x, ei)
        root = f'{ROOT_DIR}/test/dataset/test_HashedDynamicDataset'
        hash_name = f'{root}{split}_hashcache.pt'
        cards_name = f'{root}{split}_cardcache.pt'
        eh = ElphHashes(self.args)
        if os.path.exists(hash_name) and os.path.exists(cards_name):
            hashes = torch.load(hash_name)
            cards = torch.load(cards_name)
        else:
            hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
            torch.save(hashes, hash_name)
            torch.save(cards, cards_name)
        all_edges = torch.cat([self.pos_edges, self.neg_edges], 0)
        # construct features directly from hashes and cards
        subgraph_features = eh.get_subgraph_features(all_edges, hashes, cards)
        # construct features implicitly (hopefully) using the same hashes and cards
        hdd = HashDataset(root, split, data, self.pos_edges, self.neg_edges, self.args, use_coalesce=False,
                          directed=False)
        self.assertTrue(hdd.links.shape == (2 * self.n_edges, 2))
        self.assertTrue(len(hdd.labels) == 2 * self.n_edges)
        self.assertTrue(len(hdd.edge_weight) == self.edge_index.shape[1])

        dl = DataLoader(hdd, batch_size=1,
                        shuffle=False, num_workers=1)
        # check the dataset has the same features
        for sf, elem in zip(subgraph_features, dl):
            sf_test = elem[0]
            self.assertTrue(torch.all(torch.eq(sf, sf_test)))

    def test_get_subgraph_features_batched(self):
        batch_size = 3
        torch.manual_seed(0)
        self.args.model = 'BUDDY'
        split = 'test'
        ei = self.edge_index
        root = f'{ROOT_DIR}/test/dataset/test_HashedDynamicDataset'
        hash_name = f'{root}{split}_hashcache.pt'
        cards_name = f'{root}{split}_cardcache.pt'
        eh = ElphHashes(self.args)
        if os.path.exists(hash_name) and os.path.exists(cards_name):
            hashes = torch.load(hash_name)
            cards = torch.load(cards_name)
        else:
            hashes, cards = eh.build_hash_tables(self.n_nodes, self.edge_index)
            torch.save(hashes, hash_name)
            torch.save(cards, cards_name)
        all_edges = torch.cat([self.pos_edges, self.neg_edges], 0)
        subgraph_features = eh.get_subgraph_features(all_edges, hashes, cards, batch_size=batch_size)
        self.assertTrue(
            subgraph_features.shape == (len(all_edges), self.args.max_hash_hops * (self.args.max_hash_hops + 2)))
        sf2 = eh.get_subgraph_features(all_edges, hashes, cards)
        self.assertTrue(torch.all(torch.eq(subgraph_features, sf2)))

    def test_preprocess_features(self):
        pass

    def test_read_subgraph_features(self):
        pass

    def test_preprocess_subgraph_features(self):
        root = f'{ROOT_DIR}/test/dataset/test_HashedDynamicDataset'
        split = 'train'
        ei = self.edge_index
        data = Data(self.x, ei)
        hdd = HashDataset(root, split, data, self.pos_edges, self.neg_edges, self.args, use_coalesce=False,
                          directed=False,
                          load_features=True)

    def test_make_train_eval_dataset(self):
        self.args.max_hash_hops = 2
        torch.manual_seed(0)
        n_pos_samples = 8
        negs_per_pos = 10
        n_pos_edges = 10
        pos_edges = torch.randint(self.n_nodes, (n_pos_edges, 2))
        neg_edges = get_same_source_negs(self.n_nodes, negs_per_pos, pos_edges.t()).t()
        # neg_edges = torch.randint(self.n_nodes, (n_pos_edges * negs_per_pos, 2))
        split = 'train'
        ei = self.edge_index
        data = Data(self.x, ei)
        root = f'{ROOT_DIR}/test/dataset/test_HashedDynamicDataset'
        self.args.dataset_name = 'test_dataset'
        hdd = HashDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                          directed=False)
        train_eval_dataset = make_train_eval_data(self.args, hdd, self.n_nodes, n_pos_samples=n_pos_samples,
                                                  negs_per_pos=negs_per_pos)
        self.assertTrue(len(train_eval_dataset.links) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.labels) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.subgraph_features) == (negs_per_pos + 1) * n_pos_samples)
        self.args.use_RA = True
        hdd = HashDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                          directed=False)
        train_eval_dataset = make_train_eval_data(self.args, hdd, self.n_nodes, n_pos_samples=n_pos_samples,
                                                  negs_per_pos=negs_per_pos)
        self.assertTrue(len(train_eval_dataset.RA) == (negs_per_pos + 1) * n_pos_samples)
        self.args.max_hash_hops = 3
        hdd = HashDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                          directed=False)
        train_eval_dataset = make_train_eval_data(self.args, hdd, self.n_nodes, n_pos_samples=n_pos_samples,
                                                  negs_per_pos=negs_per_pos)
        self.assertTrue(len(train_eval_dataset.links) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.labels) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.subgraph_features) == (negs_per_pos + 1) * n_pos_samples)
        self.args.use_RA = True
        hdd = HashDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                          directed=False)
        train_eval_dataset = make_train_eval_data(self.args, hdd, self.n_nodes, n_pos_samples=n_pos_samples,
                                                  negs_per_pos=negs_per_pos)
        self.assertTrue(len(train_eval_dataset.RA) == (negs_per_pos + 1) * n_pos_samples)
        # citation eval is with mrr against same source negatives. The code checks if the name starts wiht 'ogbl-citation'
        # and if so, generates the same source negatives
        hdd = HashDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                          directed=True)
        train_eval_dataset = make_train_eval_data(self.args, hdd, self.n_nodes, n_pos_samples=n_pos_samples,
                                                  negs_per_pos=negs_per_pos)
        self.assertTrue(len(train_eval_dataset.links) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.labels) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.subgraph_features) == (negs_per_pos + 1) * n_pos_samples)
