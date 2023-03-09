"""
Baseline GNN models
"""

import torch
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d as BN
from torch_sparse import SparseTensor


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, residual):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, root_weight=residual))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, root_weight=residual))
        self.convs.append(SAGEConv(hidden_channels, out_channels, root_weight=residual))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SIGNBaseClass(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGNBaseClass, self).__init__()

        self.K = K
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(self.K + 1):
            self.lins.append(Linear(in_channels, hidden_channels))
            self.bns.append(BN(hidden_channels))
        self.lin_out = Linear((K + 1) * hidden_channels, out_channels)
        self.dropout = dropout
        self.adj_t = None

    def reset_parameters(self):
        for lin, bn in zip(self.lins, self.bns):
            lin.reset_parameters()
            bn.reset_parameters()

    def cache_adj_t(self, edge_index, num_nodes):
        row, col = edge_index
        adj_t = SparseTensor(row=col, col=row,
                             sparse_sizes=(num_nodes, num_nodes))

        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    def forward(self, *args):
        raise NotImplementedError


class SIGNEmbedding(SIGNBaseClass):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGNEmbedding, self).__init__(in_channels, hidden_channels, out_channels, K, dropout)

    def forward(self, x, adj_t, num_nodes):
        if self.adj_t is None:
            self.adj_t = self.cache_adj_t(adj_t, num_nodes)
        hs = []
        for lin, bn in zip(self.lins, self.bns):
            h = lin(x)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
            x = self.adj_t @ x
        h = torch.cat(hs, dim=-1)
        x = self.lin_out(h)
        return x


class SIGN(SIGNBaseClass):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGN, self).__init__(in_channels, hidden_channels, out_channels, K, dropout)

    def forward(self, xs):
        """
        apply the sign feature transform where each component of the polynomial A^n x is treated independently
        @param xs: [batch_size, 2, n_features * (K + 1)]
        @return: [batch_size, 2, hidden_dim]
        """
        xs = torch.tensor_split(xs, self.K + 1, dim=-1)
        hs = []
        # split features into k+1 chunks and put each tensor in a list
        for lin, bn, x in zip(self.lins, self.bns, xs):
            h = lin(x)
            # the next line is a fuggly way to apply the same batch norm to both source and destination edges
            h = torch.cat((bn(h[:, 0, :]).unsqueeze(1), bn(h[:, 1, :]).unsqueeze(1)), dim=1)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        x = self.lin_out(h)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
