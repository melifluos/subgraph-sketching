"""
tests for the BUDDY model
"""

import unittest
from argparse import Namespace
import os

from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.utils.random import barabasi_albert_graph
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import torch
import scipy.sparse as ssp

from src.runners.inference import get_buddy_preds
from src.runners.run import run, select_model
from src.runners.train import train_buddy
from test_params import OPT, setup_seed
from src.models.elph import BUDDY
from src.datasets.elph import get_src_dst_degree, HashDataset
from src.data import get_data, get_hashed_train_val_test_datasets, get_loaders
from src.utils import ROOT_DIR


class BUDDYTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        degree = 5
        self.num_features = 3
        self.n_edges = 10  # number of positive training edges
        self.x = torch.rand((self.n_nodes, self.num_features))
        edge_index = barabasi_albert_graph(self.n_nodes, degree)
        self.edge_index = to_undirected(edge_index)
        # self.edge_index, _ = add_self_loops(edge_index)
        edge_weight = torch.ones(self.edge_index.size(1), dtype=torch.int)
        self.A = ssp.csr_matrix((edge_weight, (self.edge_index[0], self.edge_index[1])),
                                shape=(self.n_nodes, self.n_nodes))
        self.args = Namespace(**OPT)
        self.args.model = 'BUDDY'
        setup_seed(0)
        indices = torch.randperm(self.edge_index.shape[1])[:self.n_edges]
        self.pos_edges = self.edge_index[:, indices].T
        self.neg_edges = torch.randint(self.n_nodes, (self.n_edges, 2))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_propagate(self):
        batch_size = 10
        num_features = self.x.shape[1]
        hash_hops = 2
        args = self.args
        args.sign_k = 0
        args.max_hash_hops = hash_hops
        sgf = torch.rand((batch_size, hash_hops * (hash_hops + 2)))
        gnn = BUDDY(args, num_features=num_features)
        x = torch.rand((batch_size, 2, self.num_features))
        x = gnn(sgf, x)
        self.assertTrue(len(x) == batch_size)
        # todo finish tests

    def test_degrees(self):
        src = 1
        dst = 3
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, max_nodes=1000)
        degrees = torch.tensor(self.A.sum(axis=0, dtype=float)).flatten()
        self.assertTrue(src_degree == degrees[src])
        self.assertTrue(dst_degree == degrees[dst])

    def test_get_buddy_preds(self):
        opt = {'dataset_name': 'Cora', 'cache_subgraph_features': True, 'bulk_test': True}
        opt = {**OPT, **opt}
        args = Namespace(
            **opt)
        dataset, splits, _, _ = get_data(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
        # make a model
        model = BUDDY(args, dataset.num_features)
        # make a hashing dataset
        train_dataset, val_dataset, test_dataset = get_hashed_train_val_test_datasets(dataset, train_data, val_data,
                                                                                      test_data, args)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        pos_pred, neg_pred, pred, labels = get_buddy_preds(model, test_loader, device, args=args, split='test')
        self.assertTrue(len(pos_pred) + len(neg_pred) == len(test_dataset.links))

    def test_train_buddy(self):
        opt = {'dataset_name': 'Cora', 'cache_subgraph_features': True, 'use_RA': True}
        opt = {**OPT, **opt}
        args = Namespace(
            **opt)
        dataset, splits, _, _ = get_data(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']

        # make a model
        model = BUDDY(args, dataset.num_features)
        # make a hashing dataset
        train_dataset, val_dataset, test_dataset = get_hashed_train_val_test_datasets(dataset, train_data, val_data,
                                                                                      test_data, args)
        link_loader = DataLoader(range(len(train_dataset.links)), args.batch_size, shuffle=True)
        for batch_count, indices in enumerate(link_loader):
            curr_links = torch.tensor(train_dataset.links[indices])
            subgraph_features = train_dataset.subgraph_features[indices].to(device)
            node_features = train_dataset.x[curr_links].to(device)
            degrees = train_dataset.degrees[curr_links].to(device)
            if args.use_RA:
                RA = train_dataset.RA[indices].to(device)
            else:
                RA = None
            logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA)
            self.assertTrue(len(logits), args.batch_size)
            if (batch_count + 1) * args.batch_size >= 32:
                break

        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        loss = train_buddy(model, optimizer, train_loader, args, device, emb=None)
        self.assertTrue(loss > 0)

    def test_buddy_with_sign_k(self):
        """
        test the buddy model with sign_k > 0
        Right now something doesn't make sense because the model class reads the num_features from the
        raw dataset object, but the HashedDynamicDataset has num_features * (sign_k + 1) features so it
        shouldn't even run
        :return:
        """
        self.args.sign_k = 1
        dataset, splits, _, _ = get_data(self.args)
        train_loader, train_eval_loader, val_loader, test_loader = get_loaders(self.args, dataset, splits,
                                                                               directed=False)
        model, optimizer = select_model(self.args, dataset, emb=None, device=self.device)
        loss = train_buddy(model, optimizer, train_loader, self.args, self.device)

    def test_train_with_unbiased_features(self):
        device = 'cpu'
        self.args.use_feature = False
        self.args.use_unbiased_feature = True
        self.args.batch_size = 4
        root = f'{ROOT_DIR}/test/dataset/test_train_with_unbiased_features'
        split = 'train'
        unbiased_feature_name = f'{root}_{split}_k{self.args.sign_k}_unbiased_feature_cache.pt'
        feature_name = f'{root}_{split}_k{self.args.sign_k}_featurecache.pt'
        if os.path.exists(unbiased_feature_name): os.remove(unbiased_feature_name)
        if os.path.exists(feature_name): os.remove(feature_name)
        ei = self.edge_index
        data = Data(self.x, ei)
        hdd = HashDataset(root, split, data, self.pos_edges, self.neg_edges, self.args, use_coalesce=False,
                          directed=False, load_features=True, use_unbiased_feature=True)
        model = BUDDY(self.args, self.num_features)
        link_loader = DataLoader(range(len(hdd.links)), self.args.batch_size, shuffle=True)
        for batch_count, indices in enumerate(link_loader):
            curr_links = torch.tensor(hdd.links[indices])
            subgraph_features = hdd.subgraph_features[indices].to(device)
            node_features = hdd.x[curr_links].to(device)
            unbiased_features = hdd.unbiased_features[indices].to(device)
            x = model(subgraph_features, node_features, unbiased_features=unbiased_features)
            self.assertTrue(len(x), self.args.batch_size)
            break
        if os.path.exists(unbiased_feature_name): os.remove(unbiased_feature_name)
        if os.path.exists(feature_name): os.remove(feature_name)
        # now repeat with sign_k = 0
        self.args.sign_k = 0
        hdd = HashDataset(root, split, data, self.pos_edges, self.neg_edges, self.args, use_coalesce=False,
                          directed=False, load_features=True, use_unbiased_feature=True)
        model = BUDDY(self.args, self.num_features)
        link_loader = DataLoader(range(len(hdd.links)), self.args.batch_size, shuffle=True)
        for batch_count, indices in enumerate(link_loader):
            curr_links = torch.tensor(hdd.links[indices])
            subgraph_features = hdd.subgraph_features[indices].to(device)
            node_features = hdd.x[curr_links].to(device)
            unbiased_features = hdd.unbiased_features[indices].to(device)
            x = model(subgraph_features, node_features, unbiased_features=unbiased_features)
            self.assertTrue(len(x), self.args.batch_size)
            break
        unbiased_feature_name = f'{root}_{split}_k{self.args.sign_k}_unbiased_feature_cache.pt'
        feature_name = f'{root}_{split}_k{self.args.sign_k}_featurecache.pt'
        if os.path.exists(unbiased_feature_name): os.remove(unbiased_feature_name)
        if os.path.exists(feature_name): os.remove(feature_name)

    def test_feature_forward(self):
        pass

    def test_embedding_forward(self):
        pass

    def test_forward(self):
        pass

    def test_run(self):
        # no exceptions is a pass
        self.args.train_samples = 0.1
        self.args.epochs = 1
        self.args.dataset_name = 'Cora'
        run(self.args)
