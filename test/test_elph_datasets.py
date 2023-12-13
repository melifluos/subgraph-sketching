"""
Testing the subgraph sketching dataset objects
"""
import unittest
from argparse import Namespace
import os

import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.utils import to_networkx, to_undirected, from_networkx
from torch_geometric.utils.random import barabasi_albert_graph
import scipy.sparse as ssp
import networkx as nx
import matplotlib.pyplot as plt

from src.datasets.elph import HashDataset, make_train_eval_data
from test_params import OPT
from src.utils import ROOT_DIR
from src.hashing import ElphHashes
from src.bridges import find_bridges


class ELPHDatasetTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        degree = 5
        self.n_edges = 10  # number of positive training edges
        self.n_features = 2
        self.x = torch.rand((self.n_nodes, self.n_features))
        self.edge_index = barabasi_albert_graph(self.n_nodes, degree)
        self.edge_weight = torch.ones(self.edge_index.shape[1])
        self.A = ssp.csr_matrix((self.edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.n_nodes, self.n_nodes))
        indices = torch.randperm(self.edge_index.shape[1])[:self.n_edges]
        self.pos_edges = self.edge_index[:, indices].T
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
        root = f'{ROOT_DIR}/test/dataset/test_preprocess_features'
        self.args.sign_k = 1
        split = 'train'
        ei, ew = self.edge_index, self.edge_weight
        data = Data(self.x, ei)
        hdd = HashDataset(root, split, data, self.pos_edges, self.neg_edges, self.args, use_coalesce=False,
                          directed=False)
        feature_name = f'{root}_{split}_k{self.args.sign_k}_featurecache.pt'
        if os.path.exists(feature_name): os.remove(feature_name)
        x = hdd._preprocess_node_features(data, ei, ew, verbose=True)
        self.assertTrue(x.shape == (len(hdd.links), self.n_features * (self.args.sign_k + 1)))

    def test_crop_bridge_edges(self):
        root = f'{ROOT_DIR}/test/dataset/test_crop_bridge_edges'
        split = 'train'
        unbiased_feature_name = f'{root}_{split}_k{self.args.sign_k}_unbiased_feature_cache.pt'
        feature_name = f'{root}_{split}_k{self.args.sign_k}_featurecache.pt'
        if os.path.exists(unbiased_feature_name): os.remove(unbiased_feature_name)
        if os.path.exists(feature_name): os.remove(feature_name)
        self.args.use_unbiased_feature = True
        # make a graph with some bridges
        nodes = 4
        components = 4
        edge_index = []
        bridge_list = []
        for comp in range(components):
            ei = from_networkx(nx.complete_graph(nodes)).edge_index  # (2, n_edges)
            ei += comp * nodes
            # bridge between last node in this cc and first node in next cc
            bridge = (comp * nodes + nodes - 1, (comp + 1) * nodes)
            bridge_list.append(bridge)
            ei = torch.cat([ei, torch.tensor(bridge).unsqueeze(-1)], 1)
            edge_index.append(ei)
        n_nodes = components * nodes + 1
        edge_index = to_undirected(torch.cat(edge_index, -1))
        x = torch.rand((n_nodes, self.n_features))
        data = Data(x, edge_index)
        neg_edges = torch.randint(n_nodes, (10, 2))
        hdd = HashDataset(root, split, data, edge_index.T, neg_edges, self.args, use_coalesce=False,
                          directed=False)
        self.assertTrue(len(hdd.links) == len(edge_index.T) + len(neg_edges) - len(bridge_list))

    def test_find_bridges(self):
        root = f'{ROOT_DIR}/test/dataset/test_find_bridges'
        nodes = 4
        components = 4
        edge_index = []
        bridge_list = []
        for comp in range(components):
            ei = from_networkx(nx.complete_graph(nodes)).edge_index  # (2, n_edges)
            ei += comp * nodes
            # bridge between last node in this cc and first node in next cc
            bridge = (comp * nodes + nodes - 1, (comp + 1) * nodes)
            bridge_list.append(bridge)
            ei = torch.cat([ei, torch.tensor(bridge).unsqueeze(-1)], 1)
            edge_index.append(ei)
        edge_index = to_undirected(torch.cat(edge_index, -1))
        # networkx_graph = to_networkx(Data(edge_index=edge_index), to_undirected=True)
        # pos = nx.spring_layout(networkx_graph)  # Define a layout for the graph
        # nx.draw(networkx_graph, pos, with_labels=True, font_weight='bold', node_size=100, node_color='skyblue',
        #         font_color='black', font_size=10, edge_color='gray')
        # plt.show()
        bridges = find_bridges(edge_index, root)
        self.assertTrue(bridges.shape[0] == components)
        self.assertTrue(torch.all(torch.eq(bridges, torch.tensor(bridge_list))))
        cached_bridges = torch.load(f'{root}/bridges.pt')
        self.assertTrue(torch.all(torch.eq(bridges, cached_bridges)))
        if os.path.exists(f'{root}/bridges'): os.remove(f'{root}/bridges')

    def test_get_unbiased_features(self):
        root = f'{ROOT_DIR}/test/dataset/test_get_unbiased_features'
        split = 'train'
        unbiased_feature_name = f'{root}_{split}_k{self.args.sign_k}_unbiased_feature_cache.pt'
        feature_name = f'{root}_{split}_k{self.args.sign_k}_featurecache.pt'
        if os.path.exists(unbiased_feature_name): os.remove(unbiased_feature_name)
        if os.path.exists(feature_name): os.remove(feature_name)
        ei = self.edge_index
        data = Data(self.x, ei)
        hdd = HashDataset(root, split, data, self.pos_edges, self.neg_edges, self.args, use_coalesce=False,
                          directed=False, load_features=True, use_unbiased_feature=True)
        unbiased_features = hdd._preprocess_unbiased_node_features(hdd, ei, self.edge_weight)
        os.remove(unbiased_feature_name)
        self.assertTrue(unbiased_features.shape == (len(hdd.links), self.n_features * (self.args.sign_k + 1)))
        self.assertTrue(unbiased_features.shape[0] == len(self.pos_edges) + len(self.neg_edges))

        hdd = HashDataset(root, split, data, self.pos_edges, self.neg_edges, self.args, use_coalesce=False,
                          directed=False, load_features=True, use_unbiased_feature=True)
        ubf = hdd.unbiased_features
        self.assertTrue(torch.all(torch.eq(ubf, unbiased_features)))
        self.assertTrue(ubf.shape == (len(hdd.links), self.n_features * (self.args.sign_k + 1)))
        self.assertTrue(ubf.shape[0] == len(self.pos_edges) + len(self.neg_edges))
        # check new file is made
        self.args.sign_k = 1
        hdd = HashDataset(root, split, data, self.pos_edges, self.neg_edges, self.args, use_coalesce=False,
                          directed=False, load_features=True, use_unbiased_feature=True)
        ubf = hdd.unbiased_features
        self.assertTrue(ubf.shape == (len(hdd.links), self.n_features * (self.args.sign_k + 1)))
        self.assertTrue(ubf.shape[0] == len(self.pos_edges) + len(self.neg_edges))
        # clean up
        new_feature_name = f'{root}_{split}_k{self.args.sign_k}_unbiased_feature_cache.pt'
        os.remove(new_feature_name)
        if os.path.exists(unbiased_feature_name): os.remove(unbiased_feature_name)
        if os.path.exists(feature_name): os.remove(feature_name)

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
        pos_edges = torch.randint(self.n_nodes, (10, 2))
        neg_edges = torch.randint(self.n_nodes, (10, 2))
        n_pos_samples = 8
        negs_per_pos = 10
        split = 'train'
        ei = self.edge_index
        data = Data(self.x, ei)
        root = f'{ROOT_DIR}/test/dataset/test_HashedDynamicDataset'
        hdd = HashDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                          directed=False)
        train_eval_dataset = make_train_eval_data(hdd, self.n_nodes, n_pos_samples=n_pos_samples)
        self.assertTrue(len(train_eval_dataset.links) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.labels) == (negs_per_pos + 1) * n_pos_samples)
        self.assertTrue(len(train_eval_dataset.subgraph_features) == (negs_per_pos + 1) * n_pos_samples)
        self.args.use_RA = True
        hdd = HashDataset(root, split, data, pos_edges, neg_edges, self.args, use_coalesce=False,
                          directed=False)
        train_eval_dataset = make_train_eval_data(hdd, n_pos_samples=n_pos_samples)
        self.assertTrue(len(train_eval_dataset.RA) == (negs_per_pos + 1) * n_pos_samples)
