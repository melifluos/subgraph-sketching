"""
The ELPH model
"""

from time import time
import logging

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops

from src.models.gnn import SIGN, SIGNEmbedding
from src.hashing import ElphHashes

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class LinkPredictor(torch.nn.Module):
    def __init__(self, args, use_embedding=False):
        super(LinkPredictor, self).__init__()
        self.use_embedding = use_embedding
        self.use_feature = args.use_feature
        self.feature_dropout = args.feature_dropout
        self.label_dropout = args.label_dropout
        self.dim = args.max_hash_hops * (args.max_hash_hops + 2)
        self.label_lin_layer = Linear(self.dim, self.dim)
        if args.use_feature:
            self.bn_feats = torch.nn.BatchNorm1d(args.hidden_channels)
        if self.use_embedding:
            self.bn_embs = torch.nn.BatchNorm1d(args.hidden_channels)
        self.bn_labels = torch.nn.BatchNorm1d(self.dim)
        if args.use_feature:
            self.lin_feat = Linear(args.hidden_channels,
                                   args.hidden_channels)
            self.lin_out = Linear(args.hidden_channels, args.hidden_channels)
        out_channels = self.dim + args.hidden_channels if self.use_feature else self.dim
        if self.use_embedding:
            self.lin_emb = Linear(args.hidden_channels,
                                  args.hidden_channels)
            self.lin_emb_out = Linear(args.hidden_channels, args.hidden_channels)
            out_channels += args.hidden_channels
        self.lin = Linear(out_channels, 1)

    def feature_forward(self, x):
        """
        small neural network applied edgewise to hadamard product of node features
        @param x: node features torch tensor [batch_size, 2, hidden_dim]
        @return: torch tensor [batch_size, hidden_dim]
        """
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_out(x)
        x = self.bn_feats(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return x

    def embedding_forward(self, x):
        x = self.lin_emb(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_emb_out(x)
        x = self.bn_embs(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)

        return x

    def forward(self, sf, node_features, emb=None):
        sf = self.label_lin_layer(sf)
        sf = self.bn_labels(sf)
        sf = F.relu(sf)
        x = F.dropout(sf, p=self.label_dropout, training=self.training)
        # process node features
        if self.use_feature:
            node_features = self.feature_forward(node_features)
            x = torch.cat([x, node_features.to(torch.float)], 1)
        if emb is not None:
            node_embedding = self.embedding_forward(emb)
            x = torch.cat([x, node_embedding.to(torch.float)], 1)
        x = self.lin(x)
        return x

    def print_params(self):
        print(f'model bias: {self.lin.bias.item():.3f}')
        print('model weights')
        for idx, elem in enumerate(self.lin.weight.squeeze()):
            if idx < self.dim:
                print(f'{self.idx_to_dst[idx % self.emb_dim]}: {elem.item():.3f}')
            else:
                print(f'feature {idx - self.dim}: {elem.item():.3f}')


class ELPH(torch.nn.Module):
    """
    propagating hashes, features and degrees with message passing
    """

    def __init__(self, args, num_features, node_embedding=None):
        super(ELPH, self).__init__()
        # hashing things
        self.elph_hashes = ElphHashes(args)
        self.init_hashes = None
        self.init_hll = None
        self.num_perm = args.minhash_num_perm
        self.hll_size = 2 ^ args.hll_p
        # gnn things
        self.use_feature = args.use_feature
        self.feature_prop = args.feature_prop  # None, residual, cat
        self.node_embedding = node_embedding
        self.propagate_embeddings = args.propagate_embeddings
        self.sign_k = args.sign_k
        self.label_dropout = args.label_dropout
        self.feature_dropout = args.feature_dropout
        self.num_layers = args.max_hash_hops
        self.dim = args.max_hash_hops * (args.max_hash_hops + 2)
        # construct the nodewise NN components
        self._convolution_builder(num_features, args.hidden_channels,
                                  args)  # build the convolutions for features and embs
        # construct the edgewise NN components
        self.predictor = LinkPredictor(args, node_embedding is not None)
        if self.sign_k != 0 and self.propagate_embeddings:
            # only used for the ddi where nodes have no features and transductive node embeddings are needed
            self.sign_embedding = SIGNEmbedding(args.hidden_channels, args.hidden_channels, args.hidden_channels,
                                                args.sign_k, args.sign_dropout)

    def _convolution_builder(self, num_features, hidden_channels, args):
        self.convs = torch.nn.ModuleList()
        if args.feature_prop in {'residual', 'cat'}:  # use a linear encoder
            self.feature_encoder = Linear(num_features, hidden_channels)
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        else:
            self.convs.append(
                GCNConv(num_features, hidden_channels))
        for _ in range(self.num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        if self.node_embedding is not None:
            self.emb_convs = torch.nn.ModuleList()
            for _ in range(self.num_layers):  # assuming the embedding has hidden_channels dims
                self.emb_convs.append(GCNConv(hidden_channels, hidden_channels))

    def propagate_embeddings_func(self, edge_index):
        num_nodes = self.node_embedding.num_embeddings
        gcn_edge_index, _ = gcn_norm(edge_index, num_nodes=num_nodes)
        return self.sign_embedding(self.node_embedding.weight, gcn_edge_index, num_nodes)

    def feature_conv(self, x, edge_index, k):
        if not self.use_feature:
            return None
        out = self.convs[k - 1](x, edge_index)
        out = F.dropout(out, p=self.feature_dropout, training=self.training)
        if self.feature_prop == 'residual':
            out = x + out
        return out

    def embedding_conv(self, x, edge_index, k):
        if x is None:
            return x
        out = self.emb_convs[k - 1](x, edge_index)
        out = F.dropout(out, p=self.feature_dropout, training=self.training)
        if self.feature_prop == 'residual':
            out = x + out
        return out

    def _encode_features(self, x):
        if self.use_feature:
            x = self.feature_encoder(x)
            x = F.dropout(x, p=self.feature_dropout, training=self.training)
        else:
            x = None

        return x

    def forward(self, x, edge_index):
        """
        @param x: raw node features tensor [n_nodes, n_features]
        @param adj_t: edge index tensor [2, num_links]
        @return:
        """
        hash_edge_index, _ = add_self_loops(edge_index)  # unnormalised, but with self-loops
        # if this is the first call then initialise the minhashes and hlls - these need to be the same for every model call
        num_nodes, num_features = x.shape
        if self.init_hashes == None:
            self.init_hashes = self.elph_hashes.initialise_minhash(num_nodes).to(x.device)
        if self.init_hll == None:
            self.init_hll = self.elph_hashes.initialise_hll(num_nodes).to(x.device)
        # initialise data tensors for storing k-hop hashes
        cards = torch.zeros((num_nodes, self.num_layers))
        node_hashings_table = {}
        for k in range(self.num_layers + 1):
            logger.info(f"Calculating hop {k} hashes")
            node_hashings_table[k] = {
                'hll': torch.zeros((num_nodes, self.hll_size), dtype=torch.int8, device=edge_index.device),
                'minhash': torch.zeros((num_nodes, self.num_perm), dtype=torch.int64, device=edge_index.device)}
            start = time()
            if k == 0:
                node_hashings_table[k]['minhash'] = self.init_hashes
                node_hashings_table[k]['hll'] = self.init_hll
                if self.feature_prop in {'residual', 'cat'}:  # need to get features to the hidden dim
                    x = self._encode_features(x)

            else:
                node_hashings_table[k]['hll'] = self.elph_hashes.hll_prop(node_hashings_table[k - 1]['hll'],
                                                                          hash_edge_index)
                node_hashings_table[k]['minhash'] = self.elph_hashes.minhash_prop(node_hashings_table[k - 1]['minhash'],
                                                                                  hash_edge_index)
                cards[:, k - 1] = self.elph_hashes.hll_count(node_hashings_table[k]['hll'])
                x = self.feature_conv(x, edge_index, k)

            logger.info(f'{k} hop hash generation ran in {time() - start} s')

        return x, node_hashings_table, cards


class BUDDY(torch.nn.Module):
    """
    Scalable version of ElPH that uses precomputation of subgraph features and SIGN style propagation
    of node features
    """

    def __init__(self, args, num_features=None, node_embedding=None):
        super(BUDDY, self).__init__()

        self.use_feature = args.use_feature
        self.label_dropout = args.label_dropout
        self.feature_dropout = args.feature_dropout
        self.node_embedding = node_embedding
        self.propagate_embeddings = args.propagate_embeddings
        # using both unormalised and degree normalised counts as features, hence * 2
        self.append_normalised = args.add_normed_features
        ra_counter = 1 if args.use_RA else 0
        num_labelling_features = args.max_hash_hops * (args.max_hash_hops + 2)
        self.dim = num_labelling_features * 2 if self.append_normalised else num_labelling_features
        self.use_RA = args.use_RA
        self.sign_k = args.sign_k
        if self.sign_k != 0:
            if self.propagate_embeddings:
                # this is only used for the ddi dataset where nodes have no features and transductive node embeddings are needed
                self.sign_embedding = SIGNEmbedding(args.hidden_channels, args.hidden_channels, args.hidden_channels,
                                                    args.sign_k, args.sign_dropout)
            else:
                self.sign = SIGN(num_features, args.hidden_channels, args.hidden_channels, args.sign_k,
                                 args.sign_dropout)
        self.label_lin_layer = Linear(self.dim, self.dim)
        if args.use_feature:
            self.bn_feats = torch.nn.BatchNorm1d(args.hidden_channels)
        if self.node_embedding is not None:
            self.bn_embs = torch.nn.BatchNorm1d(args.hidden_channels)
        self.bn_labels = torch.nn.BatchNorm1d(self.dim)
        self.bn_RA = torch.nn.BatchNorm1d(1)

        if args.use_feature:
            self.lin_feat = Linear(num_features,
                                   args.hidden_channels)
            self.lin_out = Linear(args.hidden_channels, args.hidden_channels)
        hidden_channels = self.dim + args.hidden_channels if self.use_feature else self.dim
        if self.node_embedding is not None:
            self.lin_emb = Linear(args.hidden_channels,
                                  args.hidden_channels)
            self.lin_emb_out = Linear(args.hidden_channels, args.hidden_channels)
            hidden_channels += self.node_embedding.embedding_dim

        self.lin = Linear(hidden_channels + ra_counter, 1)

    def propagate_embeddings_func(self, edge_index):
        num_nodes = self.node_embedding.num_embeddings
        gcn_edge_index, _ = gcn_norm(edge_index, num_nodes=num_nodes)
        return self.sign_embedding(self.node_embedding.weight, gcn_edge_index, num_nodes)

    def _append_degree_normalised(self, x, src_degree, dst_degree):
        """
        Create a set of features that have the spirit of a cosine similarity x.y / ||x||.||y||. Some nodes (particularly negative samples)
        have zero degree
        because part of the graph is held back as train / val supervision edges and so divide by zero needs to be handled.
        Note that we always divide by the src and dst node's degrees for every node in ¬the subgraph
        @param x: unormalised features - equivalent to x.y
        @param src_degree: equivalent to sum_i x_i^2 as x_i in (0,1)
        @param dst_degree: equivalent to sum_i y_i^2 as y_i in (0,1)
        @return:
        """
        # this doesn't quite work with edge weights as instead of degree, the sum_i w_i^2 is needed,
        # but it probably doesn't matter
        normaliser = torch.sqrt(src_degree * dst_degree)
        normed_x = torch.divide(x, normaliser.unsqueeze(dim=1))
        normed_x[torch.isnan(normed_x)] = 0
        normed_x[torch.isinf(normed_x)] = 0
        return torch.cat([x, normed_x], dim=1)

    def feature_forward(self, x):
        """
        small neural network applied edgewise to hadamard product of node features
        @param x: node features torch tensor [batch_size, 2, hidden_dim]
        @return: torch tensor [batch_size, hidden_dim]
        """
        if self.sign_k != 0:
            if self.propagate_embeddings:  # used for DDI
                x = self.sign_embedding(x)
            else:
                x = self.sign(x)
        else:
            x = self.lin_feat(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_out(x)
        x = self.bn_feats(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return x

    def embedding_forward(self, x):
        x = self.lin_emb(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_emb_out(x)
        x = self.bn_embs(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)

        return x

    def forward(self, sf, node_features, src_degree=None, dst_degree=None, RA=None, emb=None):
        """
        forward pass for one batch of edges
        @param sf: subgraph features [batch_size, num_hops*(num_hops+2)]
        @param node_features: raw node features [batch_size, 2, num_features]
        @param src_degree: degree of source nodes in batch
        @param dst_degree:
        @param RA:
        @param emb:
        @return:
        """
        if self.append_normalised:
            sf = self._append_degree_normalised(sf, src_degree, dst_degree)
        x = self.label_lin_layer(sf)
        x = self.bn_labels(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.label_dropout, training=self.training)
        if self.use_feature:
            node_features = self.feature_forward(node_features)
            x = torch.cat([x, node_features.to(torch.float)], 1)
        if self.node_embedding is not None:
            node_embedding = self.embedding_forward(emb)
            x = torch.cat([x, node_embedding.to(torch.float)], 1)
        if self.use_RA:
            RA = RA.unsqueeze(-1)
            RA = self.bn_RA(RA)
            x = torch.cat([x, RA], 1)
        x = self.lin(x)
        return x

    def print_params(self):
        print(f'model bias: {self.lin.bias.item():.3f}')
        print('model weights')
        for idx, elem in enumerate(self.lin.weight.squeeze()):
            if idx < self.dim:
                print(f'{self.idx_to_dst[idx % self.emb_dim]}: {elem.item():.3f}')
            else:
                print(f'feature {idx - self.dim}: {elem.item():.3f}')
