"""
main module
"""
import argparse
import time
import warnings
from math import inf

import numpy as np
import torch
from ogb.linkproppred import Evaluator
from torch_geometric.loader import DataLoader as pygDataLoader
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

torch.set_printoptions(precision=4)
import wandb
# when generating subgraphs the supervision edge is deleted, which triggers a SparseEfficiencyWarning, but this is
# not a performance bottleneck, so suppress for now
from scipy.sparse import SparseEfficiencyWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from data import get_data, pre_propagate_features
from evaluation import evaluate_auc, evaluate_hits, evaluate_mrr
from losses import get_loss
from seal_datasets import get_train_val_test_datasets
from elph_datasets import get_hashed_train_val_test_datasets, make_train_eval_data
from elph_models import ELPH, ELP, ELPHGNN
from seal_models import (SEALAA, SEALDGCNN, SEALGCN, SEALGIN, SEALMLP, SEALPMI,
                         SEALRA, SEALSAGE, SEALRAEmbedding, SEALSIGN, SEALOneLayer)
from utils import ROOT_DIR, print_model_params, select_embedding, str2bool
from wandb_setup import initialise_wandb

def run(args):
  args = initialise_wandb(args)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if (args.max_hash_hops == 1) and (not args.use_zero_one):
    print("WARNING: (0,1) feature knock out is not supported for 1 hop. Running with all features")
  print(f"executing on {device}")
  results_list = []