"""
testing grape, which is still changing, so keeping this light
"""
import unittest

from grape import Graph
import pandas as pd
import numpy as np
from torch import tensor
import networkx as nx
from torch_geometric.utils import to_undirected, is_undirected, from_networkx


class DataTests(unittest.TestCase):
    def setUp(self):
        self.edge_index = tensor([[0, 2, 2, 1], [1, 0, 1, 2]]).t()
        grid_size = 3  # For a 4x4 grid
        self.num_nodes = grid_size ** 2
        G = nx.grid_2d_graph(grid_size, grid_size)
        data = from_networkx(G)
        self.edge_index = data.edge_index

    # def test_unbiased(self):
    #     sketching = HyperSketching(
    #         number_of_hops=2,
    #         precision=10,
    #         include_selfloops=True,
    #         bits=6,
    #         normalize=False,
    #         unbiased=True,
    #         exact=True,
    #         zero_out_differences_cardinalities=False,
    #         include_node_types=False,
    #         include_edge_types=False,
    #     )

        node_df = pd.DataFrame({'name': np.arange(self.num_nodes)})
        graph = Graph.from_pd(edges_df=pd.DataFrame(
            {'src': self.edge_index[0].cpu().numpy(), 'dst': self.edge_index[1].cpu().numpy()}),
            edge_src_column='src', edge_dst_column='dst', directed=False, nodes_df=node_df)

        # if not self.use_grape_exact:
        # sketching.fit(graph)
        #
        # edge_df = sketching.get_edge_feature_from_edge_node_ids(graph, self.edge_index[:, 0].numpy().astype(np.uint32),
        #                                                         self.edge_index[:, 1].numpy().astype(np.uint32))
        # feats = tensor(edge_df['edge_features'])
        # self.assertTrue(feats.shape == (self.edge_index.shape[1], 8))

    def test_grid(self):
        pass
# this is undirected
