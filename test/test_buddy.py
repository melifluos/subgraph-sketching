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
from torch_scatter import scatter_min

from runners.train import train_elph
from runners.inference import get_buddy_preds
from test_params import OPT, setup_seed
from models.elph import BUDDY, LinkPredictor
from utils import ROOT_DIR, select_embedding
from datasets.elph import HashedDynamicDataset
from hashing import ElphHashes
from runners.run import run
from datasets.elph import get_src_dst_degree
from data import get_data, get_hashed_train_val_test_datasets
from runners.train import train_buddy

class BUDDYTests(unittest.TestCase):
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
        # todo finish tests

    def test_degrees(self):
        src = 1
        dst = 3
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, max_nodes=1000)
        degrees = torch.tensor(self.A.sum(axis=0, dtype=float)).flatten()
        self.assertTrue(src_degree == degrees[src])
        self.assertTrue(dst_degree == degrees[dst])

    def test_get_buddy_preds(self):
        opt = {'dataset_name': 'Cora', 'cache_test_structure_features': True, 'bulk_test': True}
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
        opt = {'dataset_name': 'Cora', 'cache_train_structure_features': True, 'use_RA': True}
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
            structure_features = train_dataset.structure_features[indices].to(device)
            node_features = train_dataset.x[curr_links].to(device)
            degrees = train_dataset.degrees[curr_links].to(device)
            if args.use_RA:
                RA = train_dataset.RA[indices].to(device)
            else:
                RA = None
            logits = model(structure_features, node_features, degrees[:, 0], degrees[:, 1], RA)
            self.assertTrue(len(logits), args.batch_size)
            if (batch_count + 1) * args.batch_size >= 32:
                break

        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
        loss = train_buddy(model, optimizer, train_loader, args, device, emb=None)
        self.assertTrue(loss > 0)

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
