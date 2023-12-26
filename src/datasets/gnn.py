"""
Construct datasets for running GNN baselines
"""
import torch
import torch_sparse
from torch_sparse import coalesce
from torch_geometric.utils import to_undirected
from torch_geometric.data import Dataset
import scipy.sparse as ssp

from src.utils import get_pos_neg_edges, get_src_dst_degree


class GNNDataset(Dataset):
    """
    A class that combines propagated node features x, and subgraph features that are encoded as sketches of
    nodes k-hop neighbors
    """

    def __init__(
            self, root, split, data, pos_edges, neg_edges, args, use_coalesce=False,
            directed=False, **kwargs):
        self.split = split  # string: train, valid or test
        self.root = root
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.use_coalesce = use_coalesce
        self.directed = directed
        self.args = args
        self.use_feature = args.use_feature
        super(GNNDataset, self).__init__(root)

        self.links = torch.cat([self.pos_edges, self.neg_edges], 0)  # [n_edges, 2]
        self.labels = [1] * self.pos_edges.size(0) + [0] * self.neg_edges.size(0)
        self.x = data.x

        if self.use_coalesce:  # compress multi-edge into edge with weight
            data.edge_index, data.edge_weight = coalesce(
                data.edge_index, data.edge_weight,
                data.num_nodes, data.num_nodes)

        if 'edge_weight' in data:
            self.edge_weight = data.edge_weight.view(-1)
        else:
            self.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.int)
        if self.directed:  # make undirected graphs like citation2 directed
            print(
                f'this is a directed graph. Making the adjacency matrix undirected to propagate features and calculate subgraph features')
            self.edge_index, self.edge_weight = to_undirected(data.edge_index, self.edge_weight)
        else:
            self.edge_index = data.edge_index
        self.A = ssp.csr_matrix(
            (self.edge_weight, (self.edge_index[0], self.edge_index[1])),
            shape=(data.num_nodes, data.num_nodes)
        )

    def get_bridge_edge_mask(self, bridges, links):
        """
        get a mask of the links that are not bridges
        @param bridges: Tensor [n_bridges, 2] of bridge edges
        @param links: Tensor [n_edges, 2] of edges
        @return: Tensor [n_edges] of bools
        """
        bridge_set = set([tuple(bridge) for bridge in bridges.tolist()])
        mask = [tuple(link) not in bridge_set for link in links.tolist()]
        mask = torch.tensor(mask)
        return mask

    def remove_bridges(self) -> None:
        """
        Remove positive training edges that are bridges. This is necessary because the unbiased features are
        calculated by removing the edge from the graph and propagating the node features. If the edge is a bridge
        then the graph will be disconnected and the node features will be zero.
        """
        # todo: decide how to handle bridges
        if True:
            mask = self.get_bridge_mask_from_subgraph_features()
            # torch.save(mask, f'{self.root}/bridge_mask.pt')
            # bridges = find_bridges(self.edge_index, self.root)
            # torch.save(self.subgraph_features, f'{self.root}/subgraph_features.pt')
            n_bridges = len(self.links) - mask.sum()
            print(f'found and removing {n_bridges / 2} bridge edges from training set using subgraph features')
        else:
            try:
                bridges = torch.load(f'{self.root}/bridges.pt')
            except FileNotFoundError:
                print('no bridges found. Generating bridges')
                bridges = find_bridges(self.edge_index, self.root)
            print(f'removing {len(bridges) / 2} bridge edges from training set')
            # need to find the location of the bridges in the links tensor
            mask = self.get_bridge_edge_mask(bridges, self.links)
        self.labels = list(np.array(self.labels)[mask])
        self.links = self.links[mask]
        if self.subgraph_features is not None:
            self.subgraph_features = self.subgraph_features[mask]
        if self.use_unbiased_feature:
            self.unbiased_features = self.unbiased_features[mask]
        if self.use_RA:
            self.RA = self.RA[mask]

    def _generate_sign_features(self, data, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                                sign_k: int) -> torch.Tensor:
        """
        Generate features by preprocessing using the Scalable Inception Graph Neural Networks (SIGN) method
         https://arxiv.org/abs/2004.11198
        @param data: A pyg data object
        @param sign_k: the maximum number of times to apply the propagation operator
        @return:
        """
        try:
            num_nodes = data.x.size(0)
        except AttributeError:
            num_nodes = data.num_nodes
        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight.float(), num_nodes)
        if sign_k == 0:
            # for most datasets it works best do one step of propagation instead of using the raw features
            xs = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
        else:
            xs = [data.x]
            for _ in range(sign_k):
                x = torch_sparse.spmm(edge_index, edge_weight, data.x.shape[0], data.x.shape[0], data.x)
                xs.append(x)
            xs = torch.cat(xs, dim=-1)
        return xs

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        if self.args.use_struct_feature:
            subgraph_features = self.subgraph_features[idx]
        else:
            subgraph_features = torch.zeros(self.max_hash_hops * (2 + self.max_hash_hops))

        y = self.labels[idx]
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, None)
        node_features = torch.cat([self.x[src].unsqueeze(dim=0), self.x[dst].unsqueeze(dim=0)], dim=0)
        return subgraph_features, node_features, src_degree, dst_degree, RA, y


def get_gnn_datasets(dataset, train_data, val_data, test_data, args, directed=False):
    root = f'{dataset.root}/gnn_'
    print(f'data path: {root}')
    use_coalesce = True if args.dataset_name == 'ogbl-collab' else False
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    print(
        f'before sampling, considering a superset of {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
        f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
        f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')
    print('constructing training dataset object')
    train_dataset = GNNDataset(root, 'train', train_data, pos_train_edge, neg_train_edge, args,
                               use_coalesce=use_coalesce, directed=directed)
    print('constructing validation dataset object')
    val_dataset = GNNDataset(root, 'valid', val_data, pos_val_edge, neg_val_edge, args,
                             use_coalesce=use_coalesce, directed=directed)
    print('constructing test dataset object')
    test_dataset = GNNDataset(root, 'test', test_data, pos_test_edge, neg_test_edge, args,
                              use_coalesce=use_coalesce, directed=directed)
    return train_dataset, val_dataset, test_dataset
