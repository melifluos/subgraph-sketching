import os
from distutils.util import strtobool

import numpy as np
import scipy
import torch
from scipy.stats import sem

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def neighbors(fringe, A, outgoing=True):
  """
  Retrieve neighbours of nodes within the fringe
  @param fringe: a set of nodeIDs
  @param A: scipy csr if outgoing = True, otherwise scipy csc
  @param outgoing: Boolean
  @return:
  """
  if outgoing:
    res = set(A[list(fringe)].indices)
  else:
    res = set(A[:, list(fringe)].indices)

  return res