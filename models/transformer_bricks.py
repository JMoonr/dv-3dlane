import numpy as np
import math
import cv2
import warnings
import einops
from functools import partial

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                    build_norm_layer, xavier_init, constant_init)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                        TransformerLayerSequence,
                                        build_transformer_layer_sequence)
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                    TRANSFORMER_LAYER_SEQUENCE)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.cnn.bricks.transformer import build_attention

from .utils import inverse_sigmoid, ground2img, norm_ysteps
from .lidar_module_utils import *
from .pos_utils import build_pos_encoding_layer


def generate_ref_pt(minx, miny, maxx, maxy, z, nx, ny, device='cuda'):
    if isinstance(z, list):
        nz = z[-1]
        # minx, miny, maxx, maxy : in ground coords
        xs = torch.linspace(minx, maxx, nx, dtype=torch.float, device=device
                ).view(1, -1, 1).expand(ny, nx, nz)
        ys = torch.linspace(miny, maxy, ny, dtype=torch.float, device=device
                ).view(-1, 1, 1).expand(ny, nx, nz)
        zs = torch.linspace(z[0], z[1], nz, dtype=torch.float, device=device
                ).view(1, 1, -1).expand(ny, nx, nz)
        ref_3d = torch.stack([xs, ys, zs], dim=-1)
        ref_3d = ref_3d.flatten(1, 2)
    else:
        # minx, miny, maxx, maxy : in ground coords
        xs = torch.linspace(minx, maxx, nx, dtype=torch.float, device=device
                ).view(1, -1, 1).expand(ny, nx, 1)
        ys = torch.linspace(miny, maxy, ny, dtype=torch.float, device=device
                ).view(-1, 1, 1).expand(ny, nx, 1)
        ref_3d = F.pad(torch.cat([xs, ys], dim=-1), (0, 1), mode='constant', value=z)
    return ref_3d



@TRANSFORMER_LAYER.register_module()
class DV3DLaneDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        query = super().forward(
            query=query, key=key, value=value,
            query_pos=query_pos, key_pos=key_pos,
            attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask, **kwargs)
        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DV3DLaneTransformerDecoder(TransformerLayerSequence):
    def __init__(self,
                 *args,
                 embed_dims=None,
                 post_norm_cfg=None, # dict(type='LN'),
                 M_decay_ratio=10,
                 num_query=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 num_lidar_feat=3,
                 look_forward_twice=False,
                 return_intermediate=True,
                #  use_vismask=False,
                #  use_refpt_qpos=False,
                 refpt_qpos_encode='mlp',
                 rept_pe_cfg=dict(
                    type='mlp',
                    inc=3,
                    hidc=256*4,
                    outc=256,
                 ),
                 refpt_pe_dim='3d',
                 pc_range=None,
                 mamba_attention_cfg=None,
                 uniseq_pos_embed=False,
                 max_seq_len=20,
                 encode_last_refpt_pos_in_query=False,
                 **kwargs):
        super(DV3DLaneTransformerDecoder, self).__init__(*args, **kwargs)
        assert num_lidar_feat >= 3
        self.num_lidar_feat = num_lidar_feat
        self.look_forward_twice = look_forward_twice
        self.return_intermediate = return_intermediate

        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.anchor_y_steps = anchor_y_steps
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query

        self.embed_dims = embed_dims
        self.refpt_qpos_encode = refpt_qpos_encode
        self.refpt_pe_dim = refpt_pe_dim
        self.pc_range = pc_range

    def init_weights(self):
        super().init_weights()

    def forward(self, query, key, value,
                top_view_region=None,
                bev_h=None, bev_w=None,
                init_z=0, img_feats=None,
                lidar2img=None, pad_shape=None,
                key_pos=None, key_padding_mask=None,
                sin_embed=None, reference_points=None,
                reg_branches=None, cls_branches=None,
                query_pos=None, points=None,
                **kwargs):
        assert key_padding_mask is None

        # init pts and M to generate pos embed for key/value
        xmin = top_view_region[0]
        ymin = top_view_region[1]
        zmin = top_view_region[2]
        xmax = top_view_region[3]
        ymax = top_view_region[4]
        zmax = top_view_region[5]

        intermediate = []
        project_results = []
        outputs_classes = []
        outputs_coords = []

        last_reference_points = []

        if key_pos is not None:
            sin_embed = key_pos + sin_embed
        sin_embed = sin_embed.permute(1, 0, 2).contiguous()

        B = key.shape[1]
        last_reference_points = [reference_points]

        for layer_idx, layer in enumerate(self.layers):
            query = layer(query, key=key, value=value,
                          key_pos=sin_embed,
                          reference_points=reference_points,
                          pc_range=[xmin, ymin, zmin, xmax, ymax, zmax],
                          pad_shape=pad_shape,
                          lidar2img=lidar2img,
                          query_pos=query_pos,
                          layer_idx=layer_idx,
                          **kwargs)

            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
            query = query.permute(1, 0, 2).contiguous()
            tmp = reg_branches[layer_idx](query)

            bs = tmp.shape[0]
            # iterative update
            tmp = tmp.view(bs, self.num_query,
                self.num_anchor_per_query, -1, 3)

            reference_points = reference_points.view(
                bs, self.num_query, self.num_anchor_per_query,
                self.num_points_per_anchor, 2
            )
            reference_points = inverse_sigmoid(reference_points)
            new_reference_points = torch.stack([
                reference_points[..., 0] + tmp[..., 0],
                reference_points[..., 1] + tmp[..., 1],
            ], dim=-1)
            new_reference_points = new_reference_points.sigmoid()

            # detrex DINO vs deform-detr
            reference_points = new_reference_points.detach()
            last_reference_not_detach = inverse_sigmoid(
                last_reference_points[-1]).view(
                    bs, self.num_query, self.num_anchor_per_query,
                    self.num_points_per_anchor, 2
                )
            lftwice_refpts = torch.stack([
                last_reference_not_detach[..., 0] + tmp[..., 0],
                last_reference_not_detach[..., 1] + tmp[..., 1],
            ], dim=-1).sigmoid()

            outputs_coords.append(
                torch.cat([
                    lftwice_refpts,
                    tmp[..., -1:]], dim=-1))
            last_reference_points.append(new_reference_points)

            cls_feat = query.view(
                bs, self.num_query, self.num_anchor_per_query, -1)
            cls_feat = torch.max(cls_feat, dim=2)[0]
            outputs_class = cls_branches[layer_idx](cls_feat)

            outputs_classes.append(outputs_class)
            query = query.permute(1, 0, 2).contiguous()

        if self.return_intermediate:
            return torch.stack(intermediate).permute(0, 2, 1, 3).contiguous(), project_results, outputs_classes, outputs_coords
        else:
            return query, project_results, outputs_classes, outputs_coords


@TRANSFORMER.register_module()
class DV3DLaneTransformer(BaseModule):
    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(DV3DLaneTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.init_weights()

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @property
    def with_encoder(self):
        return hasattr(self, 'encoder') and self.encoder

    def forward(self, x, mask, query,
                query_embed, pos_embed,
                reference_points=None,
                reg_branches=None, cls_branches=None,
                spatial_shapes=None,
                level_start_index=None,
                mlvl_masks=None,
                mlvl_positional_encodings=None,
                pos_embed2d=None,
                key_pos=None,
                **kwargs):
        # assert pos_embed is None
        memory = x
        # encoder
        if hasattr(self, 'encoder') and self.encoder:
            B = x.shape[1]
            # mlvl_masks = [torch.zeros((B, *s),
            #                          dtype=torch.bool, device=x.device)
            #     for s in spatial_shapes]
            valid_ratios = torch.stack(
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
            reference_points_2d = \
                self.get_reference_points(spatial_shapes,
                                          valid_ratios,
                                          device=x.device)
            memory = self.encoder(
                query=memory,
                key=memory,
                value=memory,
                key_pos=key_pos,
                query_pos=pos_embed2d,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points_2d,
                valid_ratios=valid_ratios,
            )

        if query_embed is not None:
            query_embed = query_embed.permute(1, 0, 2).contiguous()
        if mask is not None:
            mask = mask.view(bs, -1)
        if query is not None:
            query = query.permute(1, 0, 2).contiguous()

        out_dec, project_results, outputs_classes, outputs_coords = \
            self.decoder(
                query=query,
                key=memory,
                value=memory,
                key_pos=pos_embed,
                query_pos=query_embed,
                key_padding_mask=mask.astype(torch.bool) if mask is not None else None,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
        return out_dec, project_results, \
               outputs_classes, outputs_coords