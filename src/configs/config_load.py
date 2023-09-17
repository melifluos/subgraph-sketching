import os
import argparse
from yacs.config import CfgNode as CN
# adopted from torch_geometric.graphgym.config

def set_cfg(cfg):

	# ------------------------------------------------------------------------ #
	# Basic options
	# ------------------------------------------------------------------------ #
	# Cuda device number, used for machine with multiple gpus
	cfg.device = 0
	# Whether fix the running seed to remove randomness
	cfg.seed = 0

	cfg.dataset = CN()
	cfg.dataset.name = 'cora'	
	# root of dataset 
	
	cfg.dataset.cora = CN()
	cfg.dataset.cora.root = '/pfs/work7/workspace/scratch/cc7738-prefeature/TAPE' 
	cfg.dataset.cora.original = cfg.dataset.cora.root + '/dataset/cora_orig/cora'
	cfg.dataset.cora.papers =  cfg.dataset.cora.root + '/dataset/cora_orig/mccallum/cora/papers'
	cfg.dataset.cora.extractions =  cfg.dataset.cora.root + '/dataset/cora_andrew_mccallum/extractions/'
	# ------------------------------------------------------------------------ #
	return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args([])
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
