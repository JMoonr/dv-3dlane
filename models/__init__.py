from mmcv.utils import Config, DictAction
import copy

from .mmpt_encoder import CustomizedHardVFE
from .fusion_layer import Point2ImageFusion
from .backbone import PillarResNet34
from .neck import RPNG
from .model.pt_branch import Pillar3DDetector
from .head import *
from .registry import READERS
from .readers import *
from .dv3dlane import DV3DLane
# from .lidar_module_new import DualModalityKMeansMHA_New
# from .mamba_modules import MambaNxNSA_LA1Version_HQuery


# def build_latrmm(cfg_path, args):

#     if isinstance(cfg_path, str):
#         cfg = Config.fromfile(cfg_path)
#     else:
#         cfg = cfg_path

#     args = copy.deepcopy(args)
#     cfg.merge_from_dict(args)
#     model = LanePETR(cfg)

#     return model


# __all__ = ['build_latrmm']
