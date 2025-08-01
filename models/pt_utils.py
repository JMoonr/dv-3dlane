import torch
from torch import nn
import torch.nn.functional as F
from mmcv.ops import Voxelization
from .ms2one import build_ms2one

import warnings

import mmdet3d
from mmseg.models.builder import build_loss, build_head
from mmdet3d.ops import Voxelization
from mmdet3d.models import build_backbone, build_neck, builder
from mmdet3d.models.detectors import Base3DDetector, MVXTwoStageDetector
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.necks import PointNetFPNeck
from mmdet3d.models.detectors import VoxelNet
from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from mmseg.models.losses import accuracy

    
@DETECTORS.register_module()
class Custom3DDetector(MVXTwoStageDetector):
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 ms2one=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 seg_bev=False,
                 outc=256,
                 num_classes=2,
                #  seg_bev_loss_weight=1.0,
                 loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                 mask_seg_loss=False,
                 sampler=None,
                 align_corners=False,
                 loss_seg_bev=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 seg_bev_pred_layer=None,
                 init_cfg=[
                    dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                    dict(type='Xavier', layer='Linear', distribution='uniform')
                ],
                 ignore_index=255,
                 ):
        super(Custom3DDetector, self).__init__(init_cfg=init_cfg)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(
                pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        if ms2one is not None:
            self.ms2one = build_ms2one(ms2one)
        else:
            self.ms2one = None

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = builder.build_head(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = builder.build_head(img_roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')

        if self.with_img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated '
                              'key, please consider using init_cfg.')
                self.img_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)
        if self.with_img_roi_head:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated '
                              'key, please consider using init_cfg.')
                self.img_roi_head.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)
        if self.with_pts_backbone:
            if pts_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated '
                              'key, please consider using init_cfg')
                self.pts_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=pts_pretrained)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # self.seg_bev_loss_weight = seg_bev_loss_weight
        if seg_bev:
            if seg_bev_pred_layer is None:
                self.seg_bev_pred_layer = nn.Sequential(
                    nn.Conv2d(outc, outc // 2, kernel_size=3, bias=False, padding=1),
                    nn.BatchNorm2d(outc // 2),
                    nn.ReLU(True),
                    nn.Conv2d(outc // 2, outc // 4, kernel_size=3, bias=False, padding=1),
                    nn.BatchNorm2d(outc // 4),
                    nn.ReLU(True),
                    nn.Conv2d(outc // 4, self.num_classes, 1, bias=True),
                )
            else:
                self.seg_bev_pred_layer = build_head(seg_bev_pred_layer)

        self.mask_seg_loss = mask_seg_loss
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.align_corners = align_corners
        self.seg_bev = seg_bev
        self.init_weights()

    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract features of points."""
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features, pt2img_feats, hit_mask = self.pts_voxel_encoder(
            voxels, num_points, coors, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        if self.ms2one is not None:
            x = self.ms2one(x)
        if not isinstance(x, (list, tuple)):
            x = [x]
        return x, pt2img_feats, hit_mask

    def get_seg_bev_loss(self, point_feats, label, mask=None):
        if isinstance(point_feats, (list, tuple)):
            point_feats = point_feats[0]
        # TODO hard code here
        if not isinstance(self.seg_bev_pred_layer, nn.Sequential):
            point_feats = [point_feats]
        seg_bev_pred = self.seg_bev_pred_layer(point_feats)
        if label.ndim == 4:
            label = torch.max(label, dim=1)[1]
        if self.num_classes == 2:
            label = (label > 0).long()
        loss = self.bev_losses(seg_bev_pred, label.unsqueeze(1), mask=mask)
        return loss, seg_bev_pred

    def bev_losses(self, seg_logit, seg_label, mask=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        if mask is not None:
            mask = F.interpolate(mask, size=seg_label.shape[2:], mode='nearest')

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

            if self.mask_seg_loss and mask is not None:
                loss[loss_decode.loss_name] = (loss[loss_decode.loss_name] * mask).sum() \
                    / torch.clamp(mask.sum(), 1)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss


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


# class VoxelNetBackbone(VoxelNet):
#     def __init__(self, voxel_layer, out_channels=None):
#         nn.Module.__init__(self)
#         # self.channels = pp_model['voxel_encoder']['feat_channels'] + pp_model['backbone']['out_channels']
#         if out_channels is not None:
#             pp_model['backbone']['out_channels'] = out_channels
#         self.channels = pp_model['backbone']['out_channels']

#         self.backbone = build_backbone(pp_model['backbone'])
#         # self.neck = build_neck(pp_model['neck'])
#         self.voxel_layer = Voxelization(**voxel_layer)
#         pp_model['voxel_encoder']['voxel_size'] = voxel_layer['voxel_size']
#         pp_model['voxel_encoder']['point_cloud_range'] = voxel_layer['point_cloud_range']

#         self.voxel_encoder = builder.build_voxel_encoder(pp_model['voxel_encoder'])

#         pp_model['middle_encoder']['output_shape'] = self.voxel_layer.grid_size[[1, 0]]
#         self.middle_encoder = builder.build_middle_encoder(pp_model['middle_encoder'])
#         self.grid_size = self.voxel_layer.grid_size

#     def extract_feat(self, points):
#         """Extract features from points."""
#         voxels, num_points, coors = self.voxelize(points)
#         voxel_features = self.voxel_encoder(voxels, num_points, coors)
#         batch_size = coors[-1, 0].item() + 1
#         x = self.middle_encoder(voxel_features, coors, batch_size)

#         # ret = [x]
#         out = self.backbone(x)
#         # ret += out
#         # if self.with_neck:
#         #     x = self.neck(x)
#         return out
    
#     @torch.no_grad()
#     def voxelize(self, points):
#         """Apply dynamic voxelization to points.

#         Args:
#             points (list[torch.Tensor]): Points of each sample.

#         Returns:
#             tuple[torch.Tensor]: Concatenated points, number of points
#                 per voxel, and coordinates.
#         """
#         voxels, coors, num_points = [], [], []
#         for res in points:
#             res_voxels, res_coors, res_num_points = self.voxel_layer(res)
#             voxels.append(res_voxels)

#             # convert coordinate left-top as origin.
#             res_coors[:, 1] = -res_coors[:, 1]
#             res_coors[:, 1] = res_coors[:, 1] + self.grid_size[1] - 1

#             coors.append(res_coors)
#             num_points.append(res_num_points)
#         voxels = torch.cat(voxels, dim=0)
#         num_points = torch.cat(num_points, dim=0)
#         coors_batch = []
#         for i, coor in enumerate(coors):
#             coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
#             coors_batch.append(coor_pad)
#         coors_batch = torch.cat(coors_batch, dim=0)
#         return voxels, num_points, coors_batch

#     def forward(self, points):
#         x = self.extract_feat(points)
#         return dict(fp_xyzs=None, fp_features=x)