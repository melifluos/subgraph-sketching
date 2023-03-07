import copy
import os
import pickle

import numpy as np
import pandas as pd
import torch
from ogb.linkproppred import PygLinkPropPredDataset
import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   to_undirected)
from utils.utils import ROOT_DIR