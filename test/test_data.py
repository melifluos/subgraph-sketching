"""
testing data reader and preprocessing
"""
import unittest
import os
from argparse import Namespace

import torch
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected
from ogb.linkproppred import PygLinkPropPredDataset

from data import get_data, get_pos_neg_edges, get_ogb_train_negs, make_obg_supervision_edges, get_ogb_data
from utils.utils import ROOT_DIR
from test_params import OPT