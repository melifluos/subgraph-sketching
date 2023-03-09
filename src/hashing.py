"""
hashed based data sketching for graphs. Implemented in pytorch, but based on the datasketch library
"""
import sys
import copy
from time import time
import logging

import torch
from torch import tensor, float, zeros
from tqdm import tqdm
import numpy as np
from pandas.util import hash_array
from datasketch import MinHash, HyperLogLogPlusPlus, hyperloglog_const
from multiprocessing import Pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from utils import neighbors

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# the LABEL_LOOKUP primary key is the max hops, secondary key is an index into the feature vector and values are hops
# from nodes (u,v)
LABEL_LOOKUP = {1: {0: (1, 1), 1: (0, 1), 2: (1, 0)},
                2: {0: (1, 1), 1: (2, 1), 2: (1, 2), 3: (2, 2), 4: (0, 1), 5: (1, 0), 6: (0, 2), 7: (2, 0)},
                3: {0: (1, 1), 1: (2, 1), 2: (1, 2), 3: (2, 2), 4: (3, 1), 5: (1, 3), 6: (3, 2), 7: (2, 3), 8: (3, 3),
                    9: (0, 1), 10: (1, 0), 11: (0, 2), 12: (2, 0), 13: (0, 3), 14: (3, 0)}}