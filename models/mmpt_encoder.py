import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from mmcv.runner import force_fp32

from mmdet3d.models import builder
from mmdet3d.models.builder import VOXEL_ENCODERS
from mmdet3d.models.voxel_encoders import HardVFE
from mmdet3d.models.voxel_encoders.utils import VFELayer, get_paddings_indicator



@VOXEL_ENCODERS.register_module()
class CustomizedHardVFE(nn.Module):
    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 query_layer=None,
                 return_point_feats=False,
                 return_pt_center=False,
                 return_voxel_center=False,
                 **kwargs):
        self._with_cluster_intens = kwargs.pop('with_cluster_intens', False)
        if self._with_cluster_intens:
            in_channels += 1
        super().__init__()
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                # if fusion_layer:
                #     max_out = True
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)

        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

        self.query_layer = None
        if query_layer is not None:
            self.query_layer = builder.build_fusion_layer(query_layer)
        self.return_pt_center = return_pt_center
        self.return_voxel_center = return_voxel_center

    def forward(self,
                features,
                num_points,
                coors,
                img_feats=None,
                img_metas=None):
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = (
                features[:, :, :3].sum(dim=1, keepdim=True) /
                num_points.type_as(features).view(-1, 1, 1))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)
        
        if self._with_cluster_intens:
            points_intens_mean = (
                features[:, :, 3:4].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1))
            f_cluster_intens = features[:, :, 3:4] - points_intens_mean
            features_ls.append(f_cluster_intens)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(
                size=(features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        # N x 64 x (5 + 3 + 3)
        voxel_feats = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty.
        # Need to ensure that empty voxels remain set to zeros.
        voxel_count = voxel_feats.shape[1]
        # V x 64 <pt>
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)

        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)

        if (self.fusion_layer is not None and img_feats is not None):
            # voxel_feats = self.fusion_with_mask(features, mask, voxel_feats,
            #                                     coors, img_feats, img_metas)
            pt_center2img_feats, hit_mask = self.fuse2img_with_mask(
                points_mean, mask, voxel_feats, coors, 
                img_feats, img_metas)
        else:
            pt_center2img_feats = None
            hit_mask = None

        if self.query_layer is not None:
            img_voxel_feats = self.query_layer(
                points_mean, mask, voxel_feats, coors, 
                img_feats, img_metas)
            voxel_feats = img_voxel_feats
        return voxel_feats, pt_center2img_feats, hit_mask

    def fuse2img_with_mask(self, 
                           points_mean, 
                           mask, 
                           voxel_feats, 
                           coors, 
                           img_feats,
                           img_metas):
        """
        points_mean : V x 1 x 3
        mask: V x 64[voxel elements]
        voxel_feats : V x 64 x C
        coors
        """
        batch_size = coors[-1, 0] + 1
        points = []
        for i in range(batch_size):
            single_mask = (coors[:, 0] == i)
            points.append(
                torch.cat([
                    points_mean[single_mask].squeeze(1),
                    voxel_feats[single_mask],
                    # (voxel_feats[single_mask] * mask[single_mask].float()) \
                    # / torch.clamp(mask[single_mask].float(), 1e-5)
                ],
                dim=-1
            ))
        point_feats, hit_mask = self.fusion_layer(img_feats, points, img_metas)
        return point_feats, hit_mask
