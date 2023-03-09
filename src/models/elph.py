"""
The ELPH model
"""

from time import time
import logging

import torch
from torch.nn import Linear, ModuleList, Sequential as Seq, ReLU
import torch.nn.functional as F
import wandb
from torch_geometric.nn import global_add_pool, global_mean_pool
from models.gnn import SIGN, SIGNEmbedding
import torch_sparse
from torch_geometric.nn import GCNConv, MessagePassing
from hashing import MinhashPropagation, HllPropagation, ElphHashes
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops

from pyg_models import GCNCustomConv
from labelling_tricks import get_drnl_lookup

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class LinkPredictor(torch.nn.Module):
    def __init__(self, args, use_embedding=False):
        super(LinkPredictor, self).__init__()
        self.use_embedding = use_embedding
        self.label_features_branch = args.label_features_branch
        self.use_bn = args.use_bn
        self.use_feature = args.use_feature
        # self.feature_prop = args.feature_prop
        self.feature_dropout = args.feature_dropout
        self.label_dropout = args.label_dropout
        self.dim = args.max_hash_hops * (args.max_hash_hops + 2)
        # if self.feature_prop == 'cat':
        #   self.sign = SIGN(args.hidden_channels, args.hidden_channels, args.hidden_channels, self.num_layers,
        #                    args.sign_dropout)
        if self.label_features_branch:
            self.label_lin_layer = Linear(self.dim, self.dim)
        if self.use_bn:
            if args.use_feature:
                self.bn_feats = torch.nn.BatchNorm1d(args.hidden_channels)
            if self.use_embedding:
                self.bn_embs = torch.nn.BatchNorm1d(args.hidden_channels)
            self.bn_labels = torch.nn.BatchNorm1d(self.dim)
        if args.use_feature:
            # self.lin_feat = Linear(num_features, args.hidden_channels)
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
        # x = self.lin_feat(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_out(x)
        if self.use_bn:
            x = self.bn_feats(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return x

    def embedding_forward(self, x):
        x = self.lin_emb(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_emb_out(x)
        if self.use_bn:
            x = self.bn_embs(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)

        return x

    def forward(self, sf, node_features, emb=None):
        if self.label_features_branch:
            sf = self.label_lin_layer(sf)
            if self.use_bn:
                sf = self.bn_labels(sf)
            sf = F.relu(sf)
            x = F.dropout(sf, p=self.label_dropout, training=self.training)
        # process node features
        if self.use_feature:
            # if self.feature_prop == 'cat':
            #   node_features = self.sign(node_features)
            # edge_features = self.feature_forward(node_features)
            node_features = self.feature_forward(node_features)
            x = torch.cat([x, node_features.to(torch.float)], 1)
        if emb is not None:
            # if self.feature_prop == 'cat':
            #   emb = self.sign(emb)
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
    Efficient Link Prediction with Hashes
    """

    def __init__(self, args, num_features=None, node_embedding=None):
        super(ELPH, self).__init__()

        self.use_feature = args.use_feature
        self.dropout = args.dropout
        self.label_dropout = args.label_dropout
        self.feature_dropout = args.feature_dropout
        self.node_embedding = node_embedding
        self.propagate_embeddings = args.propagate_embeddings
        self.label_features_branch = args.label_features_branch
        self.use_bn = args.use_bn
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

        if self.label_features_branch:
            self.label_lin_layer = Linear(self.dim, self.dim)

        if self.use_bn:
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

    def append_degree_normalised(self, x, src_degree, dst_degree):
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
            x = self.sign(x)
        else:
            x = self.lin_feat(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_out(x)
        if self.use_bn:
            x = self.bn_feats(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return x

    def embedding_forward(self, x):
        x = self.lin_emb(x)
        x = x[:, 0, :] * x[:, 1, :]
        # mlp at the end
        x = self.lin_emb_out(x)
        if self.use_bn:
            x = self.bn_embs(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.feature_dropout, training=self.training)

        return x

    def forward(self, x, node_features, src_degree=None, dst_degree=None, RA=None, emb=None):
        """
        forward pass for one batch of edges
        @param x: structure features [batch_size, num_hops*(num_hops+2)]
        @param node_features: raw node features [batch_size, 2, num_features]
        @param src_degree: degree of source nodes in batch
        @param dst_degree:
        @param RA:
        @param emb:
        @return:
        """
        if self.append_normalised:
            x = self.append_degree_normalised(x, src_degree, dst_degree)
        if self.label_features_branch:
            x = self.label_lin_layer(x)
            if self.use_bn:
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
            if self.use_bn:
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
