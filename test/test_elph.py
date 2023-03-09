import unittest
from argparse import Namespace
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from datasketch import MinHash, HyperLogLogPlusPlus
import networkx as nx
from torch_scatter import scatter_min

from run import train_gnn, get_gnn_preds
from test_params import OPT, setup_seed
from models.elph import ELPHGNN, LinkPredictor
from utils import ROOT_DIR, select_embedding
from elph_datasets import HashedDynamicDataset
from hashing import ElphHashes


class ELPHTests(unittest.TestCase):
    def setUp(self):
        self.n_nodes = 30
        degree = 5
        p = 0.2
        self.num_features = 3
        self.x = torch.rand((self.n_nodes, self.num_features))
        self.G = nx.newman_watts_strogatz_graph(n=self.n_nodes, k=degree, p=p)
        edge_index = torch.tensor(np.array(self.G.edges)).T
        edge_index = to_undirected(edge_index)
        self.edge_index, _ = add_self_loops(edge_index)
        self.A = nx.adjacency_matrix(self.G)
        self.args = Namespace(**OPT)
        self.args.model = 'hashgnn'
        setup_seed(0)

    def test_propagate(self):
        num_features = self.x.shape[1]
        hash_hops = 2
        args = self.args
        args.max_hash_hops = hash_hops
        gnn = ELPHGNN(args, num_features=num_features)
        x, hashes, cards, emb = gnn(self.x, self.edge_index)
        self.assertTrue(x.shape == (self.n_nodes, args.hidden_channels))
        self.assertTrue(emb is None)
        self.assertTrue(len(hashes) == hash_hops + 1)
        self.assertTrue(cards.shape == (self.n_nodes, hash_hops))
        self.assertTrue(hashes[0]['hll'].shape == (self.n_nodes, 2 ** args.hll_p))
        self.assertTrue(hashes[0]['minhash'].shape == (self.n_nodes, args.minhash_num_perm))
        args.train_node_embedding = True
        emb = select_embedding(args, self.n_nodes, self.x.device)
        gnn = ELPHGNN(args, num_features=num_features, node_embedding=emb)
        x, hashes, cards, emb = gnn(self.x, self.edge_index)
        self.assertTrue(emb.shape == (self.n_nodes, args.hidden_channels))
        # not ideal, but currently still propagate features even when we're not using them
        args.use_feature = False
        emb = select_embedding(args, self.n_nodes, self.x.device)
        gnn = ELPHGNN(args, num_features=num_features, node_embedding=emb)
        x, hashes, cards, emb = gnn(self.x, self.edge_index)
        self.assertTrue(x is None)
        args.feature_prop = 'cat'
        emb = select_embedding(args, self.n_nodes, self.x.device)
        gnn = ELPHGNN(args, num_features=num_features, node_embedding=emb)
        x, hashes, cards, emb = gnn(self.x, self.edge_index)
        self.assertTrue(emb.shape == (self.n_nodes, (1 + gnn.num_layers) * args.hidden_channels))
        args.use_feature = True
        emb = select_embedding(args, self.n_nodes, self.x.device)
        gnn = ELPHGNN(args, num_features=num_features, node_embedding=emb)
        x, hashes, cards, emb = gnn(self.x, self.edge_index)
        self.assertTrue(emb.shape == (self.n_nodes, (1 + gnn.num_layers) * args.hidden_channels))
        self.assertTrue(x.shape == (self.n_nodes, (1 + gnn.num_layers) * args.hidden_channels))

    def test_model_forward(self):
        n_links = 10
        num_features = self.x.shape[1]
        gnn = ELPHGNN(self.args, num_features)
        x, hashes, cards, _ = gnn(self.x, self.edge_index)
        links = torch.randint(self.n_nodes, (n_links, 2))
        sf = gnn.elph_hashes.get_subgraph_features(links, hashes, cards)
        out = gnn.predictor(sf, x[links])
        self.assertTrue(len(out) == n_links)
