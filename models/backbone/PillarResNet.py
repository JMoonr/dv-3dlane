import torch
from torch import nn

from mmdet3d.models.builder import BACKBONES
from mmdet3d.models.builder import build_fusion_layer
from mmcv.cnn import build_norm_layer

from spconv.pytorch import SparseSequential, SparseConv2d, SparseReLU
from .base import Sparse2DBasicBlock, Sparse2DBasicBlockV, post_act_block_dense


@BACKBONES.register_module()
class PillarResNet18S(nn.Module):
    def __init__(self, in_channels=32, **kwargs):
        super().__init__()

        norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels, in_channels * 2, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * 2)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * 2, in_channels * 4, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 4)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 4, in_channels * 8, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 8)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.backbone_channels = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 128,
            'conv4': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
        }

    def forward(self, sp_tensor):
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }
        return backbone_features
    
    
@BACKBONES.register_module()
class PillarResNet18(nn.Module):
    def __init__(self, in_channels=32, fusion_layer=None, query_layer=None):
        super().__init__()

        if query_layer is not None:
            conv2_inc = query_layer.get(
                'img_channels', fusion_layer.get(
                    'img_channels', None
                ) if fusion_layer is not None else None
            )
            assert conv2_inc is not None
        else:
            conv2_inc = 0
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = build_fusion_layer(fusion_layer)
        self.query_layer = None
        if query_layer is not None:
            self.query_layer = build_fusion_layer(query_layer)

        dense_block = post_act_block_dense

        norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels + conv2_inc, in_channels * 2, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * 2)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * 2, in_channels * 4, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 4)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 4, in_channels * 8, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 8)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 128,
            'conv4': 256,
            'conv5': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
            'conv5': 16,
        }

    # def forward(self, sp_tensor):
    #     x_conv1 = self.conv1(sp_tensor)
    #     x_conv2 = self.conv2(x_conv1)
    #     x_conv3 = self.conv3(x_conv2)
    #     x_conv4 = self.conv4(x_conv3)
    #     x_conv4 = x_conv4.dense()
    #     x_conv5 = self.conv5(x_conv4)
        
    #     backbone_features = {
    #         'conv1': x_conv1,
    #         'conv2': x_conv2,
    #         'conv3': x_conv3,
    #         'conv4': x_conv4,
    #         'conv5': x_conv5,
    #     }
    #     return backbone_features

    def forward(self, sp_tensor, xyzs, img_feats, img_metas):
        x_conv1 = self.conv1(sp_tensor)

        pt_center2img_feats = hit_mask = None
        if self.fusion_layer is not None:
            points = []
            for b_idx in range(x_conv1.batch_size):
                mask = x_conv1.indices[:, 0] == b_idx
                points.append(
                    torch.cat([
                        xyzs[mask],
                        x_conv1.features[mask]
                    ], dim=-1)
                )

            pt_center2img_feats, hit_mask = self.fusion_layer(
                img_feats, points, img_metas)

        if self.query_layer is not None:
            img_voxel_feats = self.query_layer(
                xyzs, None,
                x_conv1.features, x_conv1.indices, 
                img_feats, img_metas)
            x_conv1 = x_conv1.replace_feature(img_voxel_feats)

        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
            'conv5': x_conv5,
        }
        return backbone_features, pt_center2img_feats, hit_mask


@BACKBONES.register_module()
class PillarResNet34S(nn.Module):
    def __init__(self, in_channels=32, **kwargs):
        super().__init__()

        norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels, in_channels * 2, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * 2)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * 2, in_channels * 4, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 4)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 4, in_channels * 8, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 8)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.backbone_channels = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 128,
            'conv4': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
        }

    def forward(self, sp_tensor):
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }
        return backbone_features
    
    
@BACKBONES.register_module()
class PillarResNet34(nn.Module):
    def __init__(self, in_channels=32, fusion_layer=None, query_layer=None):
        super().__init__()

        dense_block = post_act_block_dense

        norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        if query_layer is not None:
            conv2_inc = query_layer.get(
                'img_channels', fusion_layer.get(
                    'img_channels', None
                ) if fusion_layer is not None else None
            )
            assert conv2_inc is not None
        else:
            conv2_inc = 0

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels + conv2_inc, in_channels * 2, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * 2)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * 2, in_channels * 4, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 4)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 4, in_channels * 8, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 8)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels * 8, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 128,
            'conv4': 256,
            'conv5': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
            'conv5': 16,
        }
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = build_fusion_layer(fusion_layer)
        self.query_layer = None
        if query_layer is not None:
            self.query_layer = build_fusion_layer(query_layer)

    def forward(self, sp_tensor, xyzs, img_feats, img_metas):
        x_conv1 = self.conv1(sp_tensor)

        pt_center2img_feats = hit_mask = None
        if self.fusion_layer is not None:
            points = []
            for b_idx in range(x_conv1.batch_size):
                mask = x_conv1.indices[:, 0] == b_idx
                points.append(
                    torch.cat([
                        xyzs[mask],
                        x_conv1.features[mask]
                    ], dim=-1)
                )

            pt_center2img_feats, hit_mask = self.fusion_layer(
                img_feats, points, img_metas)

        if self.query_layer is not None:
            img_voxel_feats = self.query_layer(
                xyzs, None,
                x_conv1.features, x_conv1.indices, 
                img_feats, img_metas)
            x_conv1 = x_conv1.replace_feature(img_voxel_feats)

        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
            'conv5': x_conv5,
        }
        return backbone_features, pt_center2img_feats, hit_mask



@BACKBONES.register_module()
class PillarResNet50(nn.Module):
    def __init__(self, in_channels=32, fusion_layer=None, query_layer=None):
        super().__init__()

        dense_block = post_act_block_dense
        cm = 4 * 2

        norm_cfg = dict(type="BN1d", momentum=0.01, eps=1e-3)

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        if query_layer is not None:
            conv2_inc = query_layer.get(
                'img_channels', fusion_layer.get(
                    'img_channels', None
                ) if fusion_layer is not None else None
            )
            assert conv2_inc is not None
        else:
            conv2_inc = 0

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels + conv2_inc, in_channels * cm, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * cm)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * cm, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * cm, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * cm, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * cm, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * cm, in_channels * 2 * cm, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 2 * cm)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2 * cm, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 2 * cm, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 2 * cm, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 2 * cm, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 2 * cm, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 2 * cm, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 2 * cm, in_channels * 4 * cm, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 4 * cm)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4 * cm, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 4 * cm, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 4 * cm, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels * 4 * cm, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'conv1': 32 // 2 * cm,
            'conv2': 64 // 2 * cm,
            'conv3': 12 // 2 * cm,
            'conv4': 256 // 2 * cm,
            'conv5': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
            'conv5': 16,
        }
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = build_fusion_layer(fusion_layer)
        self.query_layer = None
        if query_layer is not None:
            self.query_layer = build_fusion_layer(query_layer)

    def forward(self, sp_tensor, xyzs, img_feats, img_metas):
        x_conv1 = self.conv1(sp_tensor)

        pt_center2img_feats = hit_mask = None
        if self.fusion_layer is not None:
            points = []
            for b_idx in range(x_conv1.batch_size):
                mask = x_conv1.indices[:, 0] == b_idx
                points.append(
                    torch.cat([
                        xyzs[mask],
                        x_conv1.features[mask]
                    ], dim=-1)
                )

            pt_center2img_feats, hit_mask = self.fusion_layer(
                img_feats, points, img_metas)

        if self.query_layer is not None:
            img_voxel_feats = self.query_layer(
                xyzs, None,
                x_conv1.features, x_conv1.indices, 
                img_feats, img_metas)
            x_conv1 = x_conv1.replace_feature(img_voxel_feats)

        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
            'conv5': x_conv5,
        }
        return backbone_features, pt_center2img_feats, hit_mask


