import numpy as np
import math
import cv2
import warnings

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import einops as eips
from einops import rearrange, repeat, einsum
from torch.nn.init import xavier_uniform_, constant_

from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                    TRANSFORMER_LAYER_SEQUENCE)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                    build_norm_layer, xavier_init, constant_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                        TransformerLayerSequence,
                                        build_transformer_layer_sequence,
                                        MultiheadAttention)
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.bricks.transformer import build_attention

from .utils import ground2img, SE, SE1d
from .pos_utils import build_pos_encoding_layer



@ATTENTION.register_module()
class MSDACross3DOffset(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 dropout_bev=0.1,
                 num_query=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 voxel_size=None,
                 position_range=None,
                 fusion_method='sum',
                 batch_first=False,
                 separate_attention_weights=True,
                 norm_range=False,
                 update_output_dict_query=False,
                 update_output_dict_query_method=None,
                 update_output_dict_uni_query=False,
                 update_output_dict_uni_query_method=None,
                 update_output_dict_uni_query_method_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 encode_refpt_pos=False,
                 refpt_shifted_pos_encoder=None,
                 ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False
        self.position_range = position_range
        self.voxel_size = voxel_size
        self.fusion_method = fusion_method
        self.norm_range = norm_range
        self.dropout_bev_feat = dropout_bev
        self.dropout_bev = nn.Dropout(dropout_bev)
        self.dropout = nn.Dropout(dropout)

        assert fusion_method in ['se', 'sum']
        if self.fusion_method == 'se':
            assert not batch_first
            self.fusion_module = nn.Sequential(
                SE1d(in_chnls=embed_dims * 2, ratio=4, ndim_index=0),
                nn.Linear(embed_dims * 2, embed_dims)
            )

        # for lane
        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.register_buffer('anchor_y_steps',
            torch.from_numpy(anchor_y_steps).float())
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        # uv
        self.sampling_offsets = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_points * 3 * self.num_points_per_anchor)
        self.attention_weights = nn.Linear(embed_dims,
            num_heads * num_levels * num_points * self.num_points_per_anchor)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        # bev
        # self.sampling_offsets_bev = nn.Linear(
        #     embed_dims,
        #     num_heads * num_levels * num_points * 2 * self.num_points_per_anchor)
        self.separate_attention_weights = separate_attention_weights
        if separate_attention_weights:
            self.attention_weights_bev = nn.Linear(embed_dims,
                num_heads * num_levels * num_points * self.num_points_per_anchor)
        else:
            self.modality_attention_weights = nn.Linear(
                embed_dims, num_heads * num_levels * 2)
        self.value_proj_bev = nn.Linear(embed_dims, embed_dims)
        self.output_proj_bev = nn.Linear(embed_dims, embed_dims)
        
        self.update_output_dict_query = update_output_dict_query
        self.update_output_dict_query_method = update_output_dict_query_method
        self.update_output_dict_uni_query = update_output_dict_uni_query
        self.update_output_dict_uni_query_method = update_output_dict_uni_query_method

        if self.update_output_dict_uni_query_method == 'mamba':
            assert update_output_dict_uni_query_method_cfg is not None
            self.mamba_encode_points = build_attention(update_output_dict_uni_query_method_cfg)
        else:
            self.mamba_encode_points = None

        self.encode_refpt_pos = encode_refpt_pos
        # only one of them can be True
        assert not (self.encode_refpt_pos and self.mamba_encode_points is not None)
        if self.encode_refpt_pos:
            # for encode new_refpt pos_embed (i.e., refpt + offset)
            assert refpt_shifted_pos_encoder is not None
            self.refpt_position_encoder = build_pos_encoding_layer(refpt_shifted_pos_encoder)
        else:
            self.refpt_position_encoder = None

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)

        grid_init = torch.stack([thetas.cos(), thetas.sin(), torch.zeros_like(thetas)], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 1,
            3).repeat(1, self.num_points_per_anchor, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[..., i, :] *= i + 1
        
        with torch.no_grad():
            self.sampling_offsets.bias.data = grid_init.view(-1)
        # TODO detrex use
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        
        xavier_init(self.value_proj_bev, distribution='uniform', bias=0.)
        xavier_init(self.output_proj_bev, distribution='uniform', bias=0.)

        # bev
        # constant_init(self.sampling_offsets_bev, 0.)
        # self.sampling_offsets_bev.bias.data = grid_init.view(-1)
        if self.separate_attention_weights:
            constant_init(self.attention_weights_bev, val=0., bias=0.)
        xavier_init(self.value_proj_bev, distribution='uniform', bias=0.)
        xavier_init(self.output_proj_bev, distribution='uniform', bias=0.)
        self._is_init = True

    def ref_to_lidar(self, reference_points, pc_range, not_y=True):
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        if not not_y:
            reference_points[..., 1:2] = reference_points[..., 1:2] * \
                (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
        return reference_points

    def point_sampling(self, reference_points, lidar2img, ori_shape):
        x, y, mask = ground2img(
            reference_points, H=2, W=2,
            lidar2img=lidar2img, ori_shape=ori_shape,
            mask=None, return_img_pts=True)
        return torch.stack([x, y], -1), mask

    def ref3d_to_bev(self, reference_points, position_range, voxel_size):
        x3d = reference_points[..., 0]
        y3d = reference_points[..., 1]
        return torch.stack([
            (x3d - position_range[0]) / (position_range[3] - position_range[0]),
            (y3d - position_range[1]) / (position_range[4] - position_range[1]),
        ], dim=-1)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                pc_range=None,
                lidar2img=None,
                pad_shape=None,
                key_pos=None,
                point_feats=None,
                bev_feat_flatten=None,
                bev_shapes=None,
                bev_level_start_index=None,
                sin_embed_bev=None,
                output_dict=None,
                use_refpt_pos=False,
                ref_pt3d=None,
                ref_pt3d_encoding=None,
                gen_refpt_params=None,
                **kwargs):
        # import pdb;pdb.set_trace()
        if gen_refpt_params is not None:
            assert 'query_pos' in output_dict and 'ref_pt3d' in output_dict
            query_pos = output_dict['query_pos']
            ref_pt3d = output_dict['ref_pt3d']
            reference_points = output_dict['cluster_refpts']

        if value is None:
            assert False
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if key_pos is not None:
            value = value + key_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2).contiguous()
            value = value.permute(1, 0, 2).contiguous()

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        value_bev = bev_feat_flatten
        if not self.batch_first:
            value_bev = value_bev.permute(1, 0, 2).contiguous()
        if sin_embed_bev is not None:
            value_bev = value_bev + sin_embed_bev
        value_bev = self.value_proj_bev(value_bev)
        value_bev = value_bev.view(bs, bev_feat_flatten.shape[0], self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_points_per_anchor,
            self.num_heads, self.num_levels, self.num_points, 3)
        if self.norm_range:
            assert False
            sampling_offsets = torch.cat([
                sampling_offsets[..., 0] * (pc_range[3] - pc_range[0]),
                sampling_offsets[..., 1] * (pc_range[4] - pc_range[1]),
                sampling_offsets[..., 2],
            ], dim=-1)

        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_points_per_anchor,
            self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_points_per_anchor,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        reference_points = reference_points.view(
            bs, self.num_query, self.num_anchor_per_query, -1, 2)

        if not use_refpt_pos:
            ref_pt3d = torch.cat([
                reference_points[..., 0:1], # x
                self.anchor_y_steps.view(1, 1, self.num_anchor_per_query, -1, 1
                    ).expand_as(reference_points[..., 0:1]), # y
                reference_points[..., 1:2] # z
            ], dim=-1)

        # frontview-msda
        sampling_locations = self.ref_to_lidar(ref_pt3d, pc_range, not_y=False if use_refpt_pos else True)
        # import pdb;pdb.set_trace()
        new_sampling_locations = sampling_locations.view(
                bs, self.num_query, self.num_anchor_per_query, -1, 1, 1, 1, 3) \
            + sampling_offsets.view(
                bs, self.num_query, self.num_anchor_per_query, self.num_points_per_anchor,
                *sampling_offsets.shape[3:]
            )

        new_sampling_locations = new_sampling_locations.permute(0, 1, 2, 4, 5, 6, 3, 7).contiguous()
        new_sampling_locations = new_sampling_locations.flatten(-3, -2)
        xy = 3
        num_all_points = new_sampling_locations.shape[-2]
        new_sampling_locations = new_sampling_locations.view(
            bs, num_query, self.num_heads, self.num_levels, num_all_points, xy)

        attention_weights = attention_weights.permute(0, 1, 3, 4, 5, 2).contiguous()
        attention_weights = attention_weights.flatten(-2) / self.num_points_per_anchor

        if self.separate_attention_weights:
            attention_weights_bev = self.attention_weights_bev(query).view(
                bs, num_query, self.num_points_per_anchor,
                self.num_heads, self.num_levels * self.num_points)
            attention_weights_bev = attention_weights_bev.softmax(-1)
            attention_weights_bev = attention_weights_bev.view(
                bs, num_query, self.num_points_per_anchor,
                self.num_heads, self.num_levels, self.num_points)
            attention_weights_bev = attention_weights_bev.permute(0, 1, 3, 4, 5, 2).contiguous()
            attention_weights_bev = attention_weights_bev.flatten(-2) / self.num_points_per_anchor
        else:
            modality_weights = self.modality_attention_weights(query).view(
                bs, num_query, self.num_heads, self.num_levels, 2
            )
            modality_weights = torch.softmax(modality_weights, dim=-1)
            attention_weights_bev = attention_weights * modality_weights[..., 0][..., None]
            attention_weights = attention_weights * modality_weights[..., 1][..., None]

        sampling_locations2d, mask = self.point_sampling(
            F.pad(new_sampling_locations.flatten(1, -2), (0, 1), value=1),
            lidar2img=lidar2img, ori_shape=pad_shape)
        sampling_locations2d = sampling_locations2d.view(
            *new_sampling_locations.shape[:-1], 2)

        sampling_locations_bev = self.ref3d_to_bev(
            new_sampling_locations, self.position_range, self.voxel_size)

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations2d,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations2d, attention_weights)
        
        if self.update_output_dict_query:
            if self.update_output_dict_query_method is None:
                output_dict['uv_query'] = output.view(
                    bs, self.num_query, self.num_anchor_per_query, -1).mean(dim=2)
            elif self.update_output_dict_query_method == 'sum':
                output_dict['uv_query'] =  output_dict['uv_query'] + output.view(
                    bs, self.num_query, self.num_anchor_per_query, -1).mean(dim=2)
            else:
                raise NotImplementedError()
            
        output = self.output_proj(output)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2).contiguous()
        
        output = self.dropout(output)

        if torch.cuda.is_available() and value.is_cuda:
            output_bev = MultiScaleDeformableAttnFunction.apply(
                value_bev, bev_shapes, level_start_index, sampling_locations_bev,
                attention_weights_bev, self.im2col_step)
        else:
            output_bev = multi_scale_deformable_attn_pytorch(
                value_bev, bev_shapes, sampling_locations_bev, attention_weights_bev)
        if self.update_output_dict_query:
            if self.update_output_dict_query_method is None:
                output_dict['pt_query'] = output_bev.view(
                    bs, self.num_query, self.num_anchor_per_query, -1).mean(dim=2)
            elif self.update_output_dict_query_method == 'sum':
                output_dict['pt_query'] = output_dict['pt_query'] + output_bev.view(
                    bs, self.num_query, self.num_anchor_per_query, -1).mean(dim=2)
            else:
                raise NotImplementedError()
        
        output_bev = self.output_proj_bev(output_bev)
        
        if not self.batch_first:
            output_bev = output_bev.permute(1, 0, 2).contiguous()

        if self.dropout_bev_feat > 0:
            output_bev = self.dropout_bev(output_bev)

        if self.fusion_method == 'sum':
            output = output + output_bev + identity
        elif self.fusion_method == 'se':
            output = output + identity
            output_bev = output_bev + identity
            output = self.fusion_module(
                torch.cat([output, output_bev], dim=-1))
        else:
            raise NotImplementedError()
        
        if self.encode_refpt_pos:
            refpt_pos_embed = self.refpt_position_encoder(new_sampling_locations)


        if self.update_output_dict_uni_query:
            bs_first_output = output if self.batch_first else output.permute(1, 0, 2)
            
            if self.update_output_dict_uni_query_method is None:
                bs_first_output = eips.rearrange(bs_first_output, 'b (n m) c -> b n m c', n=self.num_query)
                output_dict['query'] = bs_first_output.mean(dim=2)
            elif self.update_output_dict_uni_query_method == 'sum':
                bs_first_output = eips.rearrange(bs_first_output, 'b (n m) c -> b n m c', n=self.num_query)
                output_dict['query'] = output_dict['query'] + bs_first_output.mean(dim=2)
            elif self.update_output_dict_uni_query_method == 'mamba':
                update_key = self.mamba_encode_points.update_query_key
                mba_query, pt_mba_query, mba_query_pos = self.mamba_encode_points(
                    query=bs_first_output,
                    query_pos=eips.rearrange(query_pos, 'nm b c -> b nm c') if query_pos is not None else None,
                    mamba_seq_posemb=kwargs['mamba_seq_posemb'],
                )
                output_dict[update_key] = mba_query + output_dict[update_key]
                if self.mamba_encode_points.init_mbasa_refpos:
                    if update_key + '_pos' not in output_dict:
                        output_dict[update_key + '_pos'] = mba_query_pos
                    else:
                        output_dict[update_key + '_pos'] += mba_query_pos
            else:
                raise NotImplementedError()
            
        return output


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention Module. This code is adopted from
    https://github.com/jadore801120/attention-is-all-you-need-pytorch.

    Args:
        temperature (float): The scale factor for softmax input.
        attn_dropout (float): Dropout layer on attn_output_weights.
    """

    def __init__(
        self,
        temperature,
        attn_dropout=0.1,
        attn_dim=-1,
        gumbel_softmax=False,
        out_proj=False,
        out_dropout=0.0,
        out_residual=False,
        ):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.attn_dim = attn_dim
        self.gumbel_softmax = gumbel_softmax
        if out_proj:
            self.output_proj = nn.Linear(embed_dims, embed_dims)
        else:
            self.output_proj = None
        self.out_dropout_rate = out_dropout
        self.out_drop = nn.Dropout(out_dropout)
        self.out_residual = out_residual
    
    def init_weight(self):
        if self.output_proj is not None:
            xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self, q, k, v, mask=None, return_norm=True, tau=None,  query_pos=None, key_pos=None, residual=None):
        if query_pos is not None:
            q = q + query_pos
        if key_pos is not None:
            k = k + key_pos
        
        if residual is None:
            residual = q

        if tau is None:
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        else:
            attn = torch.matmul(q / self.temperature / tau, k.transpose(2, 3))
        # import pdb;pdb.set_trace()
        if mask is not None:
            # 1 to mask, 0 for not mask, following torch.nn.MHA
            if mask.dtype == torch.uint8:
                mask = mask.to(torch.bool)
            
            if mask.dtype == torch.bool:
                attn.masked_fill_(mask, float('-inf'))
            else:
                # float mask
                assert torch.is_floating_point(mask)
                attn += mask
            
        if self.gumbel_softmax and self.training:
            m = F.gumbel_softmax(attn, dim=self.attn_dim, hard=False)
        else:
            m = F.softmax(attn, dim=self.attn_dim)

        m1 = self.dropout(m)
        output = torch.matmul(m1, v)
        if self.output_proj is not None:
            output = self.output_proj(output)

        if self.out_dropout_rate > 0:
            output = self.out_drop(output)
        
        if self.out_residual:
            output = output + residual

        return output, m if return_norm else attn


@ATTENTION.register_module()
class DualModalityKMeansMHA(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 proj_drop=0.1,
                 attn_drop=0.1,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 num_query=None,
                 temperature=1.0,
                 temperature_scheduler=None,
                 loss='bce',
                 return_norm=True,
                 loss_weight=1.0,
                 update_query=False,
                 gumbel_softmax=False,
                 attn_dim=-2,
                 num_points=40,
                 cluster_kmeans_out_proj=False,
                 cluster_kmeans_out_drop=0.0,
                 cluster_kmeans_out_residual=False,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_heads = num_heads
        self.d_k = embed_dims // num_heads
        self.d_v = embed_dims // num_heads
        self.batch_first = batch_first
        self.loss_weight = loss_weight
        self.update_query = update_query
        self.attn_dim = attn_dim
        assert self.attn_dim == -2, 'K-MeansMHA requires -2'
        self.gumbel_softmax = gumbel_softmax

        self.embed_dims = embed_dims
        d_model = embed_dims
        qkv_bias = False

        self.dim_k = embed_dims
        self.dim_v = embed_dims

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        self.temperature_scheduler = temperature_scheduler
        if isinstance(temperature, str):
            assert temperature == 'linear'
            assert temperature_scheduler is not None
            self.temperature_scheduler = np.linspace(*temperature_scheduler)
            temperature = 1.0

        self.attention = ScaledDotProductAttention(
            temperature * self.d_k**0.5, attn_drop,
            attn_dim=attn_dim, gumbel_softmax=gumbel_softmax,
            out_proj=cluster_kmeans_out_proj,
            out_dropout=cluster_kmeans_out_drop,
            out_residual=cluster_kmeans_out_residual,
        )
        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        assert loss in ['bce', 'infonce']
        self.loss = loss
        self.return_norm = return_norm

        if loss == 'infonce':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.return_norm = False

    def forward(self,
                query=None,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                output_dict=None,
                layer_idx=None,
                y_anchor_embed=None,
                refpt_qpos_encode_func=None,
                gen_refpt_params=None,
                **kwargs):

        uv_query = output_dict.get('uv_query', None)
        pt_query = output_dict.get('pt_query', None)

        uv_query_pos = output_dict.get('uv_query_pos', None)
        pt_query_pos = output_dict.get('pt_query_pos', None)

        q_pos = k_pos = None
        q = uv_query
        k = v = pt_query

        q_pos = uv_query_pos
        k_pos = pt_query_pos

        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()

        if identity is None:
            identity = q

        q = self.linear_q(q).view(batch_size, len_q, self.num_heads, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.num_heads, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.num_heads, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if q_pos is not None:
            q_pos = q_pos.view(batch_size, len_q, self.num_heads, self.d_k).transpose(1, 2)
        if k_pos is not None:
            k_pos = k_pos.view(batch_size, len_k, self.num_heads, self.d_k).transpose(1, 2)

        if attn_mask is not None:
            assert False

        if self.temperature_scheduler is not None:
            if self.training:
                tau = self.temperature_scheduler[kwargs['epoch']]
            else:
                tau = self.temperature_scheduler[-1]
        else:
            tau = None

        attn_out, attn_map = self.attention(
            q, k, v, mask=attn_mask,
            query_pos=q_pos,
            key_pos=k_pos,
            return_norm=self.return_norm,
            tau=tau)

        if self.training:
            # deep supervision
            img_match_results = output_dict['matched_indices'][-1]
            bev_match_results = output_dict['bev_matched_indices'][-1]

            true_len_q = len_q # if not self.use_both_query else len_q // 2
            target_attn_mask = [
                torch.zeros((true_len_q, true_len_q),
                            dtype=torch.float32, device=q.device)
                for _ in range(batch_size)
            ]
            pos_attn_mask = [
                torch.zeros((true_len_q, true_len_q),
                            dtype=torch.float32, device=q.device)
                for _ in range(batch_size)
            ]
            attn_mask_loss = 0.

            for b_idx in range(batch_size):
                img_match = img_match_results[b_idx]
                bev_match = bev_match_results[b_idx]

                if len(bev_match[0]) == 0:
                    # trick to avoid training stuck
                    attn_mask_loss = attn_mask_loss + 0. * self.logit_scale
                    continue

                pos_attn_mask[b_idx][img_match[0], :] = 1
                pos_attn_mask[b_idx][:, bev_match[0]] = 1

                target_attn_mask[b_idx][
                    img_match[0],
                    torch.scatter(
                        torch.zeros_like(bev_match[0]),
                        0, bev_match[1], bev_match[0]
                    )[img_match[1]]
                ] = 1

                scale_attn_map = self.logit_scale.exp() * attn_map[b_idx]
                target_img, target_bev = torch.nonzero(target_attn_mask[b_idx], as_tuple=True)
                b_attn_mask_loss = (
                    F.cross_entropy(scale_attn_map[:, target_img, :].flatten(0, 1),
                                    target_bev.unsqueeze(0).repeat(self.num_heads, 1).flatten()) +
                    F.cross_entropy(scale_attn_map[:, :, target_bev].permute(0, 2, 1).flatten(0, 1),
                                    target_img.unsqueeze(0).repeat(self.num_heads, 1).flatten())
                ) / 2.0
                attn_mask_loss = attn_mask_loss + b_attn_mask_loss

            output_dict['l%d_attn_loss' % layer_idx] = \
                attn_mask_loss / batch_size * self.loss_weight

        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, len_q, self.dim_v)
        # B x Lq x 256
        out = self.proj_drop(self.fc(attn_out))# B x Lq x 256

        if self.update_query:
            out = identity + out
            output_dict['uv_query'] = out

        if not self.batch_first:
            out = out.transpose(0, 1)

        query = query.view(out.shape[0], -1, *query.shape[1:]) + out.unsqueeze(1)
        query = query.flatten(0, 1)

        return query
