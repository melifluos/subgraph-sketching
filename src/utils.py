import os
from distutils.util import strtobool

import numpy as np
import scipy
import torch
from scipy.stats import sem

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))