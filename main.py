import os
import os.path as osp
import argparse
# from mmcv.utils import Config, DictAction
from utils.config import Config, DictAction

import torch
# torch.autograd.set_detect_anomaly(True)

from utils.utils import *
from experiments.ddp import *
from experiments.runner import *


def get_args():
    parser = argparse.ArgumentParser()
    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local-rank", type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--use_slurm', default=False, action='store_true')

    # exp setting
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--diff_seed', action='store_true',
                        help='Whether or not set different seeds for different ranks')
    parser.add_argument('--deterministic', action='store_true',
                         help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='overwrite config param.')
    return parser.parse_args()

def main(neper=None):
    args = get_args()
    # define runner to begin training or evaluation
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # initialize distributed data parallel set
    ddp_init(args, cfg)
    cfg.merge_from_dict(vars(args))
    
    # if neper is not None:
    #     neper["parameters"] = cfg
    # import pdb;pdb.set_trace()
    runner = Runner(cfg, neper=neper)

    if not cfg.evaluate:
        print('train')
        runner.train()
    else:
        runner.eval()

    # if neper is not None:
    #     neper.stop()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork', force=True)
    # try:
    #     import neptune
    #     from neptune.types import GitRef
    #     run = neptune.init_run(
    #         git_ref=GitRef(repository_path=os.getcwd()),
    #         project="mollylulu/latrmm",
    #         api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMzFmODgyYi1hOGY1LTQzNWYtOTVmYy1mMjNmNWM0MzY1MjYifQ==",
    #         tags=["latrmm", "nnemb_uniquery_sum", "ref_pt_layer_init"]
    #     )
    # except ImportError as e:
    #     import warnings
    #     warnings.warn("'neptune' is not accessible, runner will not be logged in Neptune. Install neptune for better log.")
    #     run = None
    main()
