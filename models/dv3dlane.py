import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from mmdet3d.models import build_backbone, build_neck
from mmcv.utils import Config
from mmdet.models.builder import BACKBONES
from .dv3dlane_head import DV3DLaneHead
from .ms2one import build_ms2one
from .utils import deepFeatureExtractor_EfficientNet
from .pt_utils import Custom3DDetector


# overall network
class DV3DLane(nn.Module):
    def __init__(self, args):
        super(DV3DLane, self).__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1  # no centerline
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        self.num_proj = args.num_proj
        self.num_att = args.num_att

        _dim_ = args.dv3dlane_cfg.fpn_dim
        num_query = args.dv3dlane_cfg.num_query
        num_group = args.dv3dlane_cfg.num_group
        sparse_num_group = args.dv3dlane_cfg.sparse_num_group

        self.encoder = build_backbone(args.dv3dlane_cfg.encoder)
        self.encoder.init_weights()

        head_extra_cfgs = args.dv3dlane_cfg.get('head', {})
        assert head_extra_cfgs.get('pred_dim', self.num_y_steps) == self.num_y_steps
        head_extra_cfgs['pred_dim'] = self.num_y_steps

        # build 2d query-based instance seg
        self.head = DV3DLaneHead(
            args=args,
            dim=_dim_,
            num_group=num_group,
            num_convs=4,
            in_channels=_dim_,
            kernel_dim=_dim_,
            position_range=args.position_range,
            pos_encoding_2d=args.dv3dlane_cfg.pos_encoding_2d,
            q_pos_emb=args.dv3dlane_cfg.q_pos_emb,
            pos_encoding_bev=args.dv3dlane_cfg.pos_encoding_bev,
            num_query=num_query,
            num_classes=args.num_category,
            embed_dims=_dim_,
            transformer=args.transformer,
            sparse_ins_decoder=args.sparse_ins_decoder,
            point_backbone=getattr(args, 'point_backbone', None),
            **head_extra_cfgs,
            trans_params=args.dv3dlane_cfg.get('trans_params', {})
        )
        # self._initialize_weights(args)

    def forward(self, image, is_training=True, extra_dict=None):
        out_featList = self.encoder(image)
        extra_dict['x'] = out_featList
        output = self.head(extra_dict, is_training=is_training)
        return output