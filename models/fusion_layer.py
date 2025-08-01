import torch
from torch import nn as nn
from torch.nn import functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmcv.cnn import normal_init
from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type,
                                        points_cam2img)
from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers import apply_3d_transformation

from .utils import ground2img
from .utils import SE, get_norm

from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, PackedSequence


def point_sample(img_meta,
                 img_features,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to camera coordinate
    pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1
    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    return point_features.squeeze().t()


def point_feats2img(points_xyz,
                    img_feat_shape, 
                    lidar2img, 
                    pad_shape
                    ):
    pt2img_feats = []
    points_xyz = pad_sequence(
        points_xyz, batch_first=True, padding_value=0)

    points_xyz, points_feats = points_xyz[..., :3], points_xyz[..., 3:]
    points_xyz = F.pad(points_xyz, (0, 1), mode='constant', value=1)
    pt2img_feats = ground2img(
        points_xyz,
        *img_feat_shape,
        lidar2img,
        pad_shape,
        extra_feats=points_feats,
        with_clone=False
    ) # [:, 4:, ...]
    return pt2img_feats


@FUSION_LAYERS.register_module()
class Point2ImageFusion(BaseModule):
    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0,
                 p_drop_fuse=0.1,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 fusion_method='sigmoid_res_add',
                 lateral_conv=True):
        super().__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.fuse_no_img = fusion_method == 'only_lidar'
        if fusion_method == 'sigmoid_res_add':
            self.img_transform = nn.Sequential(
                nn.Conv2d(sum(img_channels), out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            )
        if fusion_method is None:
            # se
            self.fuse_mm_feats = nn.Sequential(
                SE(in_chnls=2 * out_channels, ratio=4),
                nn.Conv2d(2 * out_channels, out_channels, kernel_size=1),
                nn.Dropout(p=p_drop_fuse),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
            )

        self.pts_transform = nn.Sequential(
            nn.Conv2d(pts_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        )
        self.fusion_method = fusion_method
        if fusion_method == 'sigmoid_res_add':
            self.weight_img = nn.Conv2d(out_channels, 1, kernel_size=1)
            self.weight_pts = nn.Conv2d(out_channels, 1, kernel_size=1)

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=1),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.02)
                # nn.init.xavier_uniform_(m.weight)
                # # nn.init.zeros_(m.bias)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, img_feats, pts, img_metas):
        # TODO multi-level img feats & ml pt feats
        pts_feats = point_feats2img(
            points_xyz=pts,
            img_feat_shape=img_metas['img_feat_shape'],
            lidar2img=img_metas['lidar2img'],
            pad_shape=img_metas['pad_shape']
        )

        hit_mask = pts_feats[:, 3, ...]
        pts_feats = self.pts_transform(pts_feats)

        if not self.fuse_no_img:
            img_feats = self.img_transform(
                img_feats[0] if isinstance(img_feats, (list, tuple)) else img_feats)

        if self.fusion_method == 'sigmoid_res_add':
            w_img = torch.sigmoid(self.weight_img(img_feats))
            w_pts = torch.sigmoid(self.weight_pts(img_feats))
            fuse_out = fuse_out + w_img * img_feats + w_pts * pts_feats
        elif self.fusion_method is None:
            fuse_out = self.fuse_mm_feats(torch.cat([img_feats, pts_feats], dim=1))
        else:
            assert self.fusion_method == 'only_lidar'
            fuse_out = pts_feats

        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)
        return fuse_out, hit_mask

    def obtain_mlvl_feats(self, img_feats, pts, img_metas):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features
        for i in range(len(img_metas)):
            mlvl_img_feats = []
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
                                       img_metas[i]))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats, pts, img_meta):
        """Sample features from single level image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (1, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
        img_pts = point_sample(
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=pts.new_tensor(proj_mat),
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            img_crop_offset=img_crop_offset,
            img_flip=img_flip,
            img_pad_shape=img_meta['input_shape'][:2],
            img_shape=img_meta['img_shape'][:2],
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return img_pts


@FUSION_LAYERS.register_module()
class Image2PointGridSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                points_mean, 
                mask, 
                voxel_feats, 
                coors, 
                img_feats,
                img_metas):
        batch_size = coors[-1, 0] + 1
        img2point_feats = []
        lidar2img = img_metas['lidar2img']
        for b_idx in range(batch_size):
            single_mask = (coors[:, 0] == b_idx)
            xyz = points_mean[single_mask].squeeze(1)
            xyz_homo = F.pad(xyz[..., :3], (0, 1), mode='constant', value=1)
            img_pt = xyz_homo @ lidar2img[b_idx].transpose(1, 0)

            img_pt = torch.cat([
                img_pt[..., :2] / torch.maximum(
                    img_pt[..., 2:3], torch.ones_like(img_pt[..., 2:3]) * 1e-5),
                img_pt[..., 2:]
            ], dim=-1)
            img_pt = img_pt.contiguous()
            coor_y = img_pt[..., 1]
            coor_x = img_pt[..., 0]

            h, w = img_metas['pad_shape'][b_idx][:2]
            coor_y = coor_y / h * 2 - 1
            coor_x = coor_x / w * 2 - 1
            grid = torch.stack([coor_x, coor_y], dim=-1)

            mode = 'bilinear'
            align_corners = True
            padding_mode='zeros'
            point_features = F.grid_sample(
                img_feats[0][b_idx].unsqueeze(0),
                grid[None, None],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners
            )[0, :, 0, :]
            img2point_feats.append(point_features)

        img2point_feat = torch.cat(img2point_feats, dim=1).transpose(1, 0)
        return torch.cat([voxel_feats, img2point_feat], dim=1)