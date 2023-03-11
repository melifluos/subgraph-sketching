"""
configuring wandb settings
"""
import os
from copy import deepcopy

import wandb


def initialise_wandb(args, config=None):
    opt = deepcopy(vars(args))
    if config:
        opt.update(config)
    if args.wandb:
        if opt['use_wandb_offline']:
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_MODE"] = "run"
        if 'wandb_run_name' in opt.keys():
            wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                       name=opt['wandb_run_name'], reinit=True, config=opt)
        else:
            wandb.init(entity=opt['wandb_entity'], project=opt['wandb_project'], group=opt['wandb_group'],
                       reinit=True, config=opt)

        wandb.define_metric("epoch_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
        if opt['wandb_track_grad_flow']:
            wandb.define_metric("grad_flow_step")  # Customize axes - https://docs.wandb.ai/guides/track/log
            wandb.define_metric("gf_e*", step_metric="grad_flow_step")  # grad_flow_epoch*

        return wandb.config  # access all HPs through wandb.config, so logging matches execution!

    else:
        os.environ["WANDB_MODE"] = "disabled"  # sets as NOOP, saves keep writing: if opt['wandb']:
        return args
