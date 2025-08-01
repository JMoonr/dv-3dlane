import torch
from torch import nn
import torch.nn.functional as F
from mmcv.ops import Voxelization
from ..ms2one import build_ms2one

import warnings
import mmdet3d
from mmdet3d.ops import Voxelization
from mmdet3d.models import build_backbone, build_neck, builder
from mmdet3d.models.detectors import Base3DDetector, MVXTwoStageDetector
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.necks import PointNetFPNeck
from mmdet3d.models.detectors import VoxelNet

from ..readers.builder import build_reader
from ..model_init import xavier_init
from ..head.bev_head import SimpleBEVSegHead


@DETECTORS.register_module()
class Pillar3DDetector(nn.Module):
    def __init__(self,
                 reader,
                 pts_backbone,
                 pts_neck,
                 ms2one=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 pts_fusion_layer=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 seg_bev=False,
                 bev_head=None,
                 init_cfg=[
                    dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                    dict(type='Xavier', layer='Linear', distribution='uniform')
                ],
                ):
        super().__init__()
        self.seg_bev = seg_bev
        assert not seg_bev

        if reader:
            self.reader = build_reader(reader)
        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)
        else:
            self.pts_neck = None

        if seg_bev:
            head_type = bev_head.pop('type')
            self.head = SimpleBEVSegHead(**bev_head)
        else:
            self.head = None

        if ms2one is not None:
            self.ms2one = build_ms2one(ms2one)
        else:
            self.ms2one = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
       
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
    
    def extract_pts_feat(self, x, img_feats=None, img_metas=None):
        """Extract features of points."""
        hit_mask = None
        pt2img_feats = None

        x, xyzs = self.reader(x)
        x, pt2img_feats, hit_mask = self.pts_backbone(x, xyzs, img_feats, img_metas)

        if self.pts_neck is not None:
            x = self.pts_neck(x)

        x = x[::-1]
        if self.ms2one is not None:
            x = self.ms2one(x)
        if not isinstance(x, (list, tuple)):
            x = [x]
        return x, pt2img_feats, hit_mask

    def forward(self, x, **kwargs):
        if self.training:
            return self.forward_pts_train(x, **kwargs)
        else:
            return self.forward_pts_test(x)

    def forward_pts_train(self, x, 
                          gt_lanes_3d, 
                          gt_labels_seg, 
                          img_feats=None, 
                          img_metas=None):
        x, pt2img_feats, hit_mask, points_mean = self.extract_pts_feat(x)
        
        pred = self.head(x)
        losses = self.head.loss(pred, gt_labels_seg)
        return losses

    
    def forward_pts_test(self, x):
        x, pt2img_feats, hit_mask, points_mean = self.extract_pts_feat(x)
        
        pred = self.head(x)
        # TODO
        return pred


    def get_seg_bev_loss(self, seg_logits, seg_label, mask=None):
        return self.head.loss(seg_logits, seg_label)



model = dict(
    type='PointRCNN',
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=5,
        num_points=(4096, 1024, 256, 64),
        radii=((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)),
        num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
        sa_channels=(((16, 16, 32), (32, 32, 64)), ((64, 64, 128), (64, 96,
                                                                    128)),
                     ((128, 196, 256), (128, 196, 256)), ((256, 256, 512),
                                                          (256, 384, 512))),
        fps_mods=(('D-FPS'), ('D-FPS'), ('D-FPS'), ('D-FPS')),
        fps_sample_range_lists=((-1), (-1), (-1), (-1)),
        aggregation_channels=(None, None, None, None),
        dilated_group=(False, False, False, False),
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    neck=dict(
        # type='PointNetFPNeck',
        fp_channels=((1536, 512, 512), (768, 512, 512), (608, 256, 256),
                     (258, 128, 128))),
    )


class MSPointNetNeck(PointNetFPNeck):
    def forward(self, feat_dict):
        sa_xyz, sa_features = self._extract_input(feat_dict)

        fp_feature = sa_features[-1]
        fp_xyz = sa_xyz[-1]

        fp_features = []
        fp_xyzs = []

        for i in range(self.num_fp):
            # consume the points in a bottom-up manner
            fp_feature = self.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)],
                                            sa_features[-(i + 2)], fp_feature)
            fp_xyz = sa_xyz[-(i + 2)]

            fp_features.append(fp_feature)
            fp_xyzs.append(fp_xyz)
        ret = dict(fp_xyzs=fp_xyzs[::-1], fp_features=fp_features[::-1])
        return ret


class PointNetBackbone(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.backbone = build_backbone(model['backbone'])
        # self.neck = build_neck(model['neck'])
        self.neck = MSPointNetNeck(**model['neck'])

    def forward(self, points):
        pointnet_output_dict = self.backbone(points)
        neck_output_dict = self.neck(pointnet_output_dict)
        return neck_output_dict


pp_model = dict(
    type='VoxelNet',
    # voxel_layer=dict(
    #     max_num_points=32,  # max_points_per_voxel
    #     point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
    #     voxel_size=voxel_size,
    #     max_voxels=(16000, 40000)  # (training, testing) max_voxels
    # ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        # voxel_size=voxel_size,
        # point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]
    ),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64,
        # output_shape=[496, 432]
    ),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 3, 5, 5],
        layer_strides=[1, 2, 2, 2],
        out_channels=[64, 128, 256, 256]),
    # neck=dict(
    #     type='SECONDFPN',
    #     in_channels=[64, 128, 256],
    #     upsample_strides=[1, 2, 4],
    #     out_channels=[128, 128, 128]),
)
