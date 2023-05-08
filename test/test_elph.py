import unittest
from argparse import Namespace

from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.data import Data
from torch_geometric.utils.random import barabasi_albert_graph
import torch
from torch.utils.data import DataLoader
import scipy.sparse as ssp

from src.runners.train import train_elph
from src.runners.inference import get_elph_preds
from src.runners.run import run
from test_params import OPT, setup_seed
from src.models.elph import ELPH, LinkPredictor
from src.utils import ROOT_DIR, select_embedding
from src.datasets.elph import HashDataset


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
        self.args.model = 'ELPH'
        setup_seed(0)

    def test_propagate(self):
        num_features = self.x.shape[1]
        hash_hops = 2
        args = self.args
        args.max_hash_hops = hash_hops
        gnn = ELPH(args, num_features=num_features)
        x, hashes, cards = gnn(self.x, self.edge_index)
        self.assertTrue(x.shape == (self.n_nodes, args.hidden_channels))
        self.assertTrue(gnn.node_embedding is None)
        self.assertTrue(len(hashes) == hash_hops + 1)
        self.assertTrue(cards.shape == (self.n_nodes, hash_hops))
        self.assertTrue(hashes[0]['hll'].shape == (self.n_nodes, 2 ** args.hll_p))
        self.assertTrue(hashes[0]['minhash'].shape == (self.n_nodes, args.minhash_num_perm))
        args.train_node_embedding = True
        emb = select_embedding(args, self.n_nodes, self.x.device)
        gnn = ELPH(args, num_features=num_features, node_embedding=emb)
        x, hashes, cards = gnn(self.x, self.edge_index)
        self.assertTrue(gnn.node_embedding.weight.shape == (self.n_nodes, args.hidden_channels))
        # not ideal, but currently still propagate features even when we're not using them
        args.use_feature = False
        emb = select_embedding(args, self.n_nodes, self.x.device)
        gnn = ELPH(args, num_features=num_features, node_embedding=emb)
        x, hashes, cards = gnn(self.x, self.edge_index)
        self.assertTrue(x is None)
        # args.feature_prop = 'cat'
        # emb = select_embedding(args, self.n_nodes, self.x.device)
        # gnn = ELPH(args, num_features=num_features, node_embedding=emb)
        # x, hashes, cards = gnn(self.x, self.edge_index)
        # self.assertTrue(gnn.node_embedding.weight.shape == (self.n_nodes, (1 + gnn.num_layers) * args.hidden_channels))
        # args.use_feature = True
        # emb = select_embedding(args, self.n_nodes, self.x.device)
        # gnn = ELPH(args, num_features=num_features, node_embedding=emb)
        # x, hashes, cards = gnn(self.x, self.edge_index)
        # # self.assertTrue(gnn.node_embedding.weight.shape == (self.n_nodes, (1 + gnn.num_layers) * args.hidden_channels))
        # self.assertTrue(x.shape == (self.n_nodes, (1 + gnn.num_layers) * args.hidden_channels))

    def test_model_forward(self):
        n_links = 10
        num_features = self.x.shape[1]
        gnn = ELPH(self.args, num_features)
        x, hashes, cards = gnn(self.x, self.edge_index)
        links = torch.randint(self.n_nodes, (n_links, 2))
        sgf = gnn.elph_hashes.get_subgraph_features(links, hashes, cards)
        out = gnn.predictor(sgf, x[links])
        self.assertTrue(len(out) == n_links)

    def test_link_predictor(self):
        n_links = 10
        num_features = self.x.shape[1]
        args = self.args
        predictor = LinkPredictor(args)
        gnn = ELPH(args, num_features)
        x, hashes, cards = gnn(self.x, self.edge_index)
        links = torch.randint(self.n_nodes, (n_links, 2))
        sf = gnn.elph_hashes.get_subgraph_features(links, hashes, cards)
        out = gnn.predictor(sf, x[links])
        self.assertTrue(len(out) == n_links)
        out1 = predictor(sf, x[links])
        self.assertTrue(len(out1) == n_links)

    def test_train_gnn(self):
        n_links = 10
        num_features = self.x.shape[1]
        hash_hops = 2
        self.args.max_hash_hops = hash_hops
        args = self.args
        gnn = ELPH(self.args, num_features)
        parameters = list(gnn.parameters())
        device = self.x.device
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.weight_decay)
        data = Data(self.x, self.edge_index)
        root = f'{ROOT_DIR}/test/train_HashedDynamicDataset'
        split = 'train'
        pos_edges = torch.randint(self.n_nodes, (n_links, 2))
        neg_edges = torch.randint(self.n_nodes, (n_links, 2))
        hdd = HashDataset(root, split, data, pos_edges, neg_edges, args, use_coalesce=False, directed=False)
        dl = DataLoader(hdd, batch_size=1,
                        shuffle=False, num_workers=1)
        loss = train_elph(gnn, optimizer, dl, args, device)
        self.assertTrue(type(loss) == float)
        # try with embeddings
        args.train_node_embedding = True
        emb = select_embedding(args, self.n_nodes, self.x.device)
        gnn = ELPH(args, num_features, emb)
        self.assertTrue(gnn.node_embedding.weight.shape == (self.n_nodes, args.hidden_channels))
        loss = train_elph(gnn, optimizer, dl, args, device)
        self.assertTrue(type(loss) == float)
        # now check the propagation of embeddings also works
        args.propagate_embeddings = True
        gnn = ELPH(args, num_features, emb)
        loss = train_elph(gnn, optimizer, dl, args, device)
        self.assertTrue(type(loss) == float)
        # try without features
        args.use_feature = False
        gnn = ELPH(args, num_features, emb)
        loss = train_elph(gnn, optimizer, dl, args, device)
        self.assertTrue(type(loss) == float)
        # and now residual
        args.feature_prop = 'residual'
        gnn = ELPH(args, num_features, emb)
        loss = train_elph(gnn, optimizer, dl, args, device)
        self.assertTrue(type(loss) == float)
        # and jk / cat
        args.feature_prop = 'cat'
        gnn = ELPH(args, num_features, emb)
        loss = train_elph(gnn, optimizer, dl, args, device)
        self.assertTrue(type(loss) == float)

    def test_get_gnn_preds(self):
        n_links = 10
        num_features = self.x.shape[1]
        hash_hops = 2
        self.args.max_hash_hops = hash_hops
        args = self.args
        gnn = ELPH(args, num_features)
        gnn.eval()
        data = Data(self.x, self.edge_index)
        root = f'{ROOT_DIR}/test/test_HashedDynamicDataset'
        split = 'train'
        pos_edges = torch.randint(self.n_nodes, (n_links, 2))
        neg_edges = torch.randint(self.n_nodes, (n_links, 2))
        hdd = HashDataset(root, split, data, pos_edges, neg_edges, args, use_coalesce=False, directed=False)
        dl = DataLoader(hdd, batch_size=1, shuffle=False, num_workers=1)
        pos_pred, neg_pred, pred, labels = get_elph_preds(gnn, dl, self.x.device, args, split='test')
        self.assertTrue(len(pos_pred == len(pos_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges) + len(pos_edges)))
        self.assertTrue(torch.sum(labels) == len(pos_edges))
        # try with embeddings
        args.train_node_embedding = True
        emb = select_embedding(args, self.n_nodes, self.x.device)
        gnn = ELPH(args, num_features, emb)
        self.assertTrue(gnn.node_embedding.weight.shape == (self.n_nodes, args.hidden_channels))
        pos_pred, neg_pred, pred, labels = get_elph_preds(gnn, dl, self.x.device, args, split='test')
        self.assertTrue(len(pos_pred == len(pos_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges) + len(pos_edges)))
        self.assertTrue(torch.sum(labels) == len(pos_edges))
        # now check the propagation of embeddings also works
        args.propagate_embeddings = True
        gnn = ELPH(args, num_features, emb)
        pos_pred, neg_pred, pred, labels = get_elph_preds(gnn, dl, self.x.device, args, split='test')
        self.assertTrue(len(pos_pred == len(pos_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges) + len(pos_edges)))
        self.assertTrue(torch.sum(labels) == len(pos_edges))
        # w/o features
        args.use_feature = False
        gnn = ELPH(args, num_features, emb)
        pos_pred, neg_pred, pred, labels = get_elph_preds(gnn, dl, self.x.device, args, split='test')
        self.assertTrue(len(pos_pred == len(pos_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges) + len(pos_edges)))
        self.assertTrue(torch.sum(labels) == len(pos_edges))
        # residual
        args.feauture_prop = 'residual'
        gnn = ELPH(args, num_features, emb)
        pos_pred, neg_pred, pred, labels = get_elph_preds(gnn, dl, self.x.device, args, split='test')
        self.assertTrue(len(pos_pred == len(pos_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges) + len(pos_edges)))
        self.assertTrue(torch.sum(labels) == len(pos_edges))
        # cat / jk
        args.feauture_prop = 'cat'
        gnn = ELPH(args, num_features, emb)
        pos_pred, neg_pred, pred, labels = get_elph_preds(gnn, dl, self.x.device, args, split='test')
        self.assertTrue(len(pos_pred == len(pos_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges)))
        self.assertTrue(len(neg_pred == len(neg_edges) + len(pos_edges)))
        self.assertTrue(torch.sum(labels) == len(pos_edges))

    def test_run(self):
        # no exceptions is a pass
        self.args.train_samples = 0.1
        self.args.epochs = 1
        self.args.dataset_name = 'Cora'
        run(self.args)
