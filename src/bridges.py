"""
preprocssing of a graph that identifies the bridges and writes them out to file
"""
import os

import networkx as nx
from torch_geometric.utils import to_networkx, to_undirected
from torch_geometric.data import Data
import torch

from src.utils import ROOT_DIR

def find_bridges(edge_index: torch.Tensor, root: str) -> torch.Tensor:
    """
    Find the bridges in a graph and write them out to file
    @param edge_index:
    @param root: This is the root dir for the dataset e.g 'dataset/Cora
    @return:
    """
    if not os.path.exists(root):
        os.makedirs(root)
    graph = to_networkx(Data(edge_index=edge_index), to_undirected=True)
    bridges = []
    total_comps = nx.number_connected_components(graph)
    for edge in graph.edges():
        # Remove the edge and check the number of connected components
        graph.remove_edge(*edge)
        num_components = nx.number_connected_components(graph)
        graph.add_edge(*edge)  # Add the edge back

        # If the number of components increases, the edge is critical
        if num_components > total_comps:
            bridges.append(edge)
    undirected_bridges = to_undirected(torch.tensor(bridges).T).T  # [num_bridges * 2, 2]
    torch.save(undirected_bridges, f'{root}/bridges.pt')

    return undirected_bridges
