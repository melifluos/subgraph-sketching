"""
This file contains the code to generate splits that maintain graph topology

1. read data into graph
2. get lcc
3. split into train / val / test edges
. Generate negative edges
4. construct pyg data objects for each split
"""
import grape
import numpy as np
import torch
from typing import Optional, Tuple

from torch_geometric.data import Data
from grape.datasets.linqs import get_words_data, Cora, CiteSeer, PubMedDiabetes
from grape.utils.networkx_utils import convert_networkx_graph_to_ensmallen_graph, \
    convert_ensmallen_graph_to_networkx_graph


# data is [nodes, features]
# graph, data = get_words_data(Cora())
# data = data.values
#
# nccs = graph.get_number_of_connected_components()
# print(f'There are {nccs} connected components in the graph')
# graph.remove_components(top_k_components=1)
# ccs = graph.get_connected_components()


def get_graph(dataset_name):
    """
    get a graph from the grape library
    :param dataset_name: string name of the dataset
    :param seed: int random seed
    :return: a graph from the grape library
    """
    if dataset_name.lower() == 'cora':
        dataset = Cora()
    elif dataset_name.lower() == 'citeseer':
        dataset = CiteSeer()
    elif dataset_name.lower() == 'pubmed':
        dataset = PubMedDiabetes()
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    graph, data = get_words_data(dataset)

    return graph, data


def get_splits(dataset_name: str, use_lcc=True, seed=None, negs_per_pos=1) -> dict[str, Data]:
    """
    get train, val, test splits from a graph
    :param dataset_name: string name of the dataset
    :return: train, val, test splits
    """
    graph, data = get_graph(dataset_name)
    nccs = graph.get_number_of_connected_components()
    print(
        f'There are {nccs[0]} connected components in {dataset_name}. with {graph.get_number_of_nodes()} nodes '
        f'and {graph.get_number_of_directed_edges() / 2.} edges')
    if use_lcc:
        print(f'Using the largest connected component')
        graph = graph.remove_components(top_k_components=1)
        assert graph.get_number_of_connected_components()[0] == 1
        print(
            f'The lcc has {graph.get_number_of_nodes} nodes and {graph.get_number_of_directed_edges() / 2.} edges')
    x = data.loc[graph.get_node_names()].values
    edge_index = graph.get_directed_edge_node_ids()
    assert edge_index.min() == 0
    assert edge_index.max() + 1 == graph.get_number_of_nodes()
    pos = get_grape_splits(graph, seed=seed)
    neg = get_grape_neg_splits(graph, seed=seed, negs_per_pos=negs_per_pos)
    pyg_graphs = []
    for edges in zip(pos, neg):
        pyg_graphs.append(grape_graph_to_pyg_data(edge_index, edges, x))
    retval = {'train': pyg_graphs[0], 'valid': pyg_graphs[1], 'test': pyg_graphs[2]}
    return retval


def get_grape_splits(graph: grape.Graph, seed: Optional[int] = None) -> (grape.Graph, grape.Graph, grape.Graph):
    """
    get train, val, test splits from a graph
    :param graph: a graph from the grape library
    :return: train, val, test splits
    """
    train, test = graph.connected_holdout(train_size=0.8, random_state=seed)
    sub_train, val = train.connected_holdout(train_size=7. / 8., random_state=seed)
    assert graph.get_number_of_edges() == sub_train.get_number_of_edges() + val.get_number_of_edges() + test.get_number_of_edges()
    return sub_train, val, test


def get_grape_neg_splits(graph, seed=None, negs_per_pos: int = 1):
    """
    get train, val, test splits from a graph
    :param graph: a graph from the grape library
    :return: train, val, test splits
    """
    # there is a small bias here because the negatives are sampled from the distribution of the entire graph
    negative_graph = graph.sample_negative_graph(
        number_of_negative_samples=negs_per_pos * graph.get_number_of_directed_edges(), random_state=seed,
    )
    neg_train, neg_test = negative_graph.random_holdout(train_size=0.8, random_state=seed)
    sub_neg_train, neg_val = neg_train.random_holdout(train_size=7 / 8., random_state=seed)
    assert negative_graph.get_number_of_edges() == sub_neg_train.get_number_of_edges() + neg_val.get_number_of_edges() + neg_test.get_number_of_edges()
    return sub_neg_train, neg_val, neg_test

def graph_to_edge_index(graph):
    return torch.LongTensor(np.int64(graph.get_directed_edge_node_ids()))

def grape_graph_to_pyg_data(edge_index: torch.Tensor, edges: Tuple[torch.Tensor, torch.Tensor],
                            x: torch.Tensor) -> Data:
    """
    convert a graph from the grape library to a pytorch geometric data object
    @param edge_index: [2, n_edges] message passing graph
    @param edges: (pos_edges, neg_edges) positive and negative edges [n_edges, 2]
    @param x: node features [n_nodes, n_features]
    @return:
    """
    edge_index = torch.LongTensor(np.int64(edge_index.T))  # [2,n_edges] message passing graph
    x = torch.FloatTensor(x)  # [n_nodes, n_features] node features
    pos_edges = graph_to_edge_index(edges[0])  # [n_pos_edges, 2] positive edges
    neg_edges = graph_to_edge_index(edges[1])  # [n_neg_edges, 2] positive edges
    data = Data(x=x, edge_index=edge_index)
    data['edge_label_index'] = torch.cat([pos_edges, neg_edges], dim=0).T  # [2, n_supervision_edges]
    data['edge_label'] = torch.cat([torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))], dim=0)
    return data


if __name__ == "__main__":
    data = get_splits('cora')
