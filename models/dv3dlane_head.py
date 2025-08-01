import numpy as np
import math
import cv2

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.init import normal_

from mmcv.cnn import bias_init_with_prob
from mmdet.models.builder import build_loss
from mmdet.models.utils import build_transformer
from mmdet.core import multi_apply
from mmdet3d.models import build_backbone, build_neck
from mmcv.utils import Config
from mmdet3d.models import build_detector

from .ms2one import build_ms2one
from .sparse_ins import SparseInsDecoder
from .pos_utils import build_pos_encoding_layer
from .utils import inverse_sigmoid, ground2img
from .transformer_bricks import *
from .norm_utils import build_iam_norm_layer
from .depthnet import build_depth_net
from .pred_utils import build_cls_reg_heads, build_refpts_pred_layer, build_fc_layer, GFlatPredLayer


class DV3DLaneHead(nn.Module):
    def __init__(self, args,
                 dim=128,
                 num_group=1,
                 num_convs=4,
                 in_channels=128,
                 kernel_dim=128,
                 pos_encoding_2d=dict(
                    type='SinePositionalEncoding',
                    num_feats=128 // 2, normalize=True),
                 pos_encoding_bev=None,
                 q_pos_emb=dict(
                    type='LiDAR_XYZ_PE',#TODO Ray_PE
                    pos_encoding_3d=dict(
                        type='MLP',
                        in_channels=3,
                        mid_channels=128*4,
                        out_channels=128),
                    pos_emb_gen='repeat',
                 ),
                 num_classes=21,
                 num_query=30,
                 embed_dims=128,
                 transformer=None,
                 num_reg_fcs=2,
                 depth_start=3,
                 position_range=[-50, 3, -10, 50, 103, 10.],
                 pred_dim=10,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_reg=dict(type='L1Loss', loss_weight=2.0),
                 sparse_ins_decoder=Config(
                    dict(
                        encoder=dict(
                            out_dims=64),# neck output feature channels
                        decoder=dict(
                            num_group=1,
                            output_iam=True,
                            scale_factor=1.),
                        sparse_decoder_weight=1.0,
                        )),
                 neck=None,
                 ms2one=None,
                 xs_loss_weight=1.0,
                 zs_loss_weight=5.0,
                 vis_loss_weight=1.0,
                 cls_loss_weight=20,
                 project_loss_weight=1.0,
                 trans_params=dict(
                     init_z=0, bev_h=250, bev_w=100),
                 num_pt_per_line=5,
                 num_feature_levels=1,
                 num_feature_levels_bev=1,
                 gt_project_h=20,
                 gt_project_w=30,
                 num_lidar_feat=3,
                 point_backbone=None,
                 insert_lidar_feat_before_img=False,
                 depth_net=None,
                 sparse_ins_bev=None,
                 project_crit=dict(
                     type='SmoothL1Loss'),
                 share_pred_heads=True,
                 ):
        super().__init__()
        self.args = args
        self.lidar_load_multi_frame = args.lidar_load_multi_frame
        self.num_lidar_feat = num_lidar_feat
        self.insert_lidar_feat_before_img = insert_lidar_feat_before_img
        # self.key_pos_after_neck = key_pos_after_neck
        # self.exchange_query_cont_pos = exchange_query_cont_pos

        top_view_region = position_range
        self.trans_params = dict(
            top_view_region = position_range)
        self.top_view_region = top_view_region
        self.gt_project_h = gt_project_h
        self.gt_project_w = gt_project_w

        self.num_y_steps = args.num_y_steps
        self.register_buffer('anchor_y_steps',
            torch.from_numpy(args.anchor_y_steps).float())
        self.register_buffer('anchor_y_steps_dense',
            torch.from_numpy(args.anchor_y_steps_dense).float())
        project_crit_t = project_crit.pop('type')
        project_crit['reduction'] = 'none'
        self.project_crit = getattr(
            nn, project_crit_t)(**project_crit)

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        # anchor num along y-axis.
        self.code_size = pred_dim
        self.num_query = num_query
        self.num_group = num_group
        self.num_pred = transformer['decoder']['num_layers']
        self.pc_range = position_range
        self.xs_loss_weight = xs_loss_weight
        self.zs_loss_weight = zs_loss_weight
        self.vis_loss_weight = vis_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.project_loss_weight = project_loss_weight

        loss_reg['reduction'] = 'none'
        self.reg_crit = build_loss(loss_reg)
        self.cls_crit = build_loss(loss_cls)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.sparse_ins = SparseInsDecoder(cfg=sparse_ins_decoder)

        self.position_range = position_range
        self.share_pred_heads = share_pred_heads

        self.q_pos_emb_mtd = q_pos_emb
        self.adapt_pos3d = build_pos_encoding_layer(
            dict(
                type='MLP',
                in_channels=self.embed_dims,
                mid_channels=self.embed_dims * 4,
                out_channels=self.embed_dims
            ))
        self.pos_encoding_2d = build_pos_encoding_layer(pos_encoding_2d)
        # current this is None
        if pos_encoding_bev is None:
            self.pos_encoding_bev = None
            self.bev_pe = 'none'
        elif pos_encoding_bev['type'] == 'BEVSineMLP':
            self.pos_encoding_bev = build_pos_encoding_layer(pos_encoding_bev)
            self.bev_pe = 'ongoing'
        else:
            self.pos_encoding_bev = nn.Sequential(
                build_pos_encoding_layer(pos_encoding_bev), build_pos_encoding_layer(
                dict(
                    type='MLP',
                    in_channels=self.embed_dims,
                    mid_channels=self.embed_dims * 4,
                    out_channels=self.embed_dims
                )))
            self.bev_pe = 'old'

        if self.q_pos_emb_mtd['type'] == 'LiDAR_XYZ_PE':
            self.pos_encoding_3d = build_pos_encoding_layer(
                q_pos_emb['pos_encoding_3d'])
        else:
            assert self.q_pos_emb_mtd['type'] is None

        self.transformer = build_transformer(transformer)

        self.seg_bev = getattr(args, 'seg_bev', False)
        if point_backbone is not None:
            self.point_backbone = build_detector(point_backbone)
            self.bev_level_embeds = nn.Parameter(torch.Tensor(
                num_feature_levels_bev, self.embed_dims))
            normal_(self.bev_level_embeds)
        else:
            self.point_backbone = None

        self.query_embedding = build_fc_layer(self.embed_dims,
        mid_dim=self.embed_dims,
        out_dim=self.embed_dims)

        self.num_reg_fcs = num_reg_fcs
        reg_out_c = 3 * self.code_size
        reg_out_c = reg_out_c // num_pt_per_line
        self.cls_branches, self.reg_branches = build_cls_reg_heads(
            self.num_reg_fcs, self.embed_dims, self.num_classes,
            reg_out_c, self.num_pred, self.share_pred_heads
        )
        
        self.num_pt_per_line = num_pt_per_line
        self.point_embedding = nn.Embedding(
            self.num_pt_per_line, self.embed_dims)

        self.reference_points = build_refpts_pred_layer(
            self.embed_dims, 2 * self.code_size // num_pt_per_line
        )
        self.num_feature_levels = num_feature_levels
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))

        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        if ms2one is not None:
            self.ms2one = build_ms2one(ms2one)
        else:
            self.ms2one = None

        if depth_net is not None:
            self.depth_net = build_depth_net(depth_net)
        else:
            self.depth_net = None

        self.sparse_ins_bev = sparse_ins_bev
        self.sparse_ins_bev = SparseInsDecoder(cfg=sparse_ins_bev)

        self._init_weights()

    def _init_weights(self):
        self.transformer.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0)
        normal_(self.level_embeds)

    def forward(self, input_dict, is_training=True):
        output_dict = {}
        img_feats = input_dict['x']

        mf_point_feats = mf_pt2img_feats = mf_hit_mask = None
        if self.point_backbone is not None:
            if self.lidar_load_multi_frame:
                single_lidar_mask = input_dict['points'][..., -1] == 0
                single_lidar = input_dict['points'][single_lidar_mask].view(
                    single_lidar_mask.shape[0], -1, input_dict['points'].shape[-1])[..., :-1]

            point_feats, pt2img_feats, hit_mask = \
                self.point_backbone.extract_pts_feat(
                    input_dict['points'] if not self.lidar_load_multi_frame else single_lidar, 
                    img_feats=img_feats, 
                    img_metas=dict(
                        img_feat_shape=img_feats[0].shape[-2:],
                        lidar2img = input_dict['lidar2img'],
                        pad_shape = input_dict['pad_shape'],
                    ))

            # TODO hard code here
            if self.neck is not None:
                if self.insert_lidar_feat_before_img:
                    img_feats = (pt2img_feats, *img_feats)
                else:
                    img_feats = (pt2img_feats, *img_feats[1:])
        else:
            point_feats, pt2img_feats = None, None

        key_pos_output = self.get_3d_keypos(
            points=input_dict['points'] if not self.lidar_load_multi_frame else single_lidar,
            T_lidar2img=input_dict['lidar2img'],
            img_shape=img_feats[0].shape[-2:],
            pad_shape=input_dict['pad_shape'],
            img_feats=img_feats,
            point_feats=point_feats,
            hit_mask=hit_mask,
            pt2img_feats=pt2img_feats,
            mf_pt2img_feats=mf_pt2img_feats,
            mf_hit_mask=mf_hit_mask,
            x=input_dict['x'],
        )
        key_pos = key_pos_output.pop('key_pos')
        output_dict.update(key_pos_output)

        if self.neck is not None:
            img_neck_out = self.neck(img_feats)

        if self.ms2one is not None:
            img_feats = self.ms2one(img_neck_out)

        if not isinstance(img_feats, (list, tuple)):
            img_feats = [img_feats]

        sparse_bev_output = self.sparse_ins_bev(
            point_feats[0],
            pos_emb_3d=None,
            lane_idx_map=input_dict['bev_seg_idx'],
            input_shape=input_dict['bev_seg_idx_label'].shape[-2:],
            point_feats=None,
            is_training=is_training,
            pt_center_feats=None,
        )
        query_bev_pos = sparse_bev_output['inst_features']
        output_dict['pt_query'] = sparse_bev_output['inst_features']
        for k, v in sparse_bev_output.items():
            output_dict['bev_%s' % k] = v

        sparse_output = self.sparse_ins(
            img_feats[0],
            pos_emb_3d=key_pos,
            lane_idx_map=input_dict['lane_idx'],
            input_shape=input_dict['seg'].shape[-2:],
            point_feats=point_feats,
            is_training=is_training,
            pt_center_feats=pt2img_feats,
        )
        output_dict.update(sparse_output)

        # generate 2d pos emb
        B, C, H, W = img_feats[0].shape
        masks = img_feats[0].new_zeros((B, H, W))

        # TODO use actual mask if using padding or other aug
        sin_embed = self.pos_encoding_2d(masks)
        sin_embed = self.adapt_pos3d(sin_embed)

        query_img_pos = sparse_output['inst_features'] # BxNxC
        query_pos = query_img_pos.unsqueeze(2) + self.point_embedding.weight[None, None, ...]
        query_pos_embeds = self.query_embedding(query_pos).flatten(1, 2)

        reference_points = self.reference_points(query_pos_embeds)
        reference_points = reference_points.sigmoid()
        output_dict['uv_query'] = sparse_output['inst_features']
        query_cont_embeds = torch.zeros_like(query_pos_embeds)

        # pos
        assert sparse_output['iam'].shape[-1] == W
        assert sparse_output['iam'].shape[-2] == H

        mlvl_feats = img_feats
        output_dict['mlvl_feats'] = [*img_neck_out, *mlvl_feats]

        feat_flatten = []
        spatial_shapes = []
        mlvl_masks = []
        assert self.num_feature_levels == len(mlvl_feats)
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2).permute(2, 0, 1).contiguous() # NxBxC
            feat = feat + self.level_embeds[None, lvl:lvl+1, :].to(feat.device)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            mlvl_masks.append(torch.zeros((bs, *spatial_shape),
                                           dtype=torch.bool,
                                           device=feat.device))

        feat_flatten = torch.cat(feat_flatten, 0)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=query_img_pos.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1, )),
             spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        # build bev infos
        bev_shapes = []
        bev_feat_flatten = []
        sin_embed_bev = None
        if point_feats is not None:
            for lvl, bev_feat in enumerate(point_feats):
                bs, c, h, w = bev_feat.shape
                bev_shape = (h, w)
                bev_feat = bev_feat.flatten(2).permute(2, 0, 1).contiguous()
                # TODO
                bev_feat = bev_feat + self.bev_level_embeds[None, lvl:lvl+1, :].to(feat.device)
                bev_shapes.append(bev_shape)
                bev_feat_flatten.append(bev_feat)
            bev_feat_flatten = torch.cat(bev_feat_flatten, 0)
            bev_shapes = torch.as_tensor(
                bev_shapes, dtype=torch.long, device=query_img_pos.device)
            bev_level_start_index = torch.cat(
                (bev_shapes.new_zeros((1, )),
                bev_shapes.prod(1).cumsum(0)[:-1])
            )

            if self.pos_encoding_bev is not None:
                masks_bev = point_feats[0].new_zeros(
                    (B, *point_feats[0].shape[-2:]))
                sin_embed_bev = self.pos_encoding_bev(masks_bev)
                if self.bev_pe == 'old':
                    sin_embed_bev = self.adapt_pos3d_bev(sin_embed_bev)
        else:
            bev_level_start_index = None

        output_dict['bev_feats'] = point_feats

        outs_dec, _, outputs_classes, outputs_coords = \
            self.transformer(
                feat_flatten, None,
                query=query_cont_embeds, 
                query_embed=query_pos_embeds,
                pos_embed=key_pos,
                reference_points=reference_points,
                reg_branches=self.reg_branches,
                cls_branches=self.cls_branches,
                img_feats=img_feats,
                lidar2img=input_dict['lidar2img'],
                pad_shape=input_dict['pad_shape'],
                sin_embed=sin_embed,
                sin_embed_bev=sin_embed_bev,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                mlvl_masks=mlvl_masks,
                mlvl_positional_encodings=None, # mlvl_positional_encodings
                points=input_dict['points'],
                pos_embed2d=None,
                point_feats=point_feats,
                bev_feat_flatten=bev_feat_flatten,
                bev_shapes=bev_shapes,
                bev_level_start_index=bev_level_start_index,
                output_dict=output_dict,
                y_anchor_embed=self.point_embedding.weight,
                epoch=input_dict.get('epoch', None),
                **self.trans_params
            )

        all_cls_scores = torch.stack(outputs_classes)
        all_line_preds = torch.stack(outputs_coords)
        all_line_preds[..., 0] = (all_line_preds[..., 0]
            * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_line_preds[..., 1] = (all_line_preds[..., 1]
            * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        # reshape to original format
        all_line_preds = all_line_preds.view(
            len(outputs_classes), bs,
            self.num_query,
            self.transformer.decoder.num_anchor_per_query,
            self.transformer.decoder.num_points_per_anchor, 2 + 1 # xz+vis
        )
        all_line_preds = all_line_preds.permute(0, 1, 2, 5, 3, 4).contiguous()
        all_line_preds = all_line_preds.flatten(3, 5)

        output_dict.update({
            'all_cls_scores': all_cls_scores,
            'all_line_preds': all_line_preds,
        })

        if is_training:
            losses = self.get_loss(dict(
                all_line_preds=all_line_preds[..., :self.num_query, :],
                all_cls_scores=all_cls_scores[..., :self.num_query, :],
                matched_indices=output_dict['matched_indices']
            ), input_dict)
            project_results = key_pos_output.get('project_results', [])
            if len(project_results):
                project_loss = self.get_project_loss(
                    project_results, input_dict,
                    h=self.gt_project_h, w=self.gt_project_w)
                losses['project_loss'] = \
                    self.project_loss_weight * project_loss
            output_dict.update(losses)

            if self.depth_net is not None:
                depth_out = self.depth_net(
                img_feats,
                input_dict,
                return_gt=True
            )
            output_dict['depth_pred'] = depth_out['depth_pred']
            if 'depth_loss' in depth_out:
                output_dict['depth_loss'] = depth_out['depth_loss']
            output_dict['depth_gt'] = depth_out['depth_gt']
            output_dict['depth_fg_mask'] = depth_out['fg_mask']

        return output_dict

    def get_project_loss(self, results, input_dict, h=20, w=30):
        gt_lane = input_dict['ground_lanes_dense']
        gt_ys = self.anchor_y_steps_dense.clone()
        code_size = gt_ys.shape[0]
        gt_xs = gt_lane[..., :code_size]
        gt_zs = gt_lane[..., code_size : 2*code_size]
        gt_vis = gt_lane[..., 2*code_size:3*code_size]
        gt_ys = gt_ys[None, None, :].expand_as(gt_xs)
        gt_points = torch.stack([gt_xs, gt_ys, gt_zs], dim=-1)

        B = results[0].shape[0]
        ref_3d_home = F.pad(gt_points, (0, 1), value=1)

        coords_img = ground2img(
            ref_3d_home,
            h, w,
            input_dict['lidar2img'],
            input_dict['pad_shape'], mask=gt_vis)

        all_loss = 0.
        for projct_result in results:
            projct_result = F.interpolate(
                projct_result,
                size=(h, w),
                mode='nearest')
            gt_proj = coords_img.clone()

            mask = (gt_proj[:, -1, ...] > 0) * (projct_result[:, -1, ...] > 0)
            diff_loss = self.project_crit(
                projct_result[:, :3, ...],
                gt_proj[:, :3, ...],
            )
            diff_y_loss = diff_loss[:, 1, ...]
            diff_z_loss = diff_loss[:, 2, ...]
            diff_loss = diff_y_loss * 0.1 + diff_z_loss

            diff_loss = (diff_loss * mask).sum() / torch.clamp(mask.sum(), 1)
            all_loss = all_loss + diff_loss

        return all_loss / len(results)

    def get_loss(self, output_dict, input_dict):
        all_cls_pred = output_dict['all_cls_scores']
        all_lane_pred = output_dict['all_line_preds']
        gt_lanes = input_dict['ground_lanes']
        all_xs_loss = 0.0
        all_zs_loss = 0.0
        all_vis_loss = 0.0
        all_cls_loss = 0.0

        matched_indices = output_dict['matched_indices']
        num_layers = all_lane_pred.shape[0]

        def single_layer_loss(layer_idx):
            gcls_pred = all_cls_pred[layer_idx]
            glane_pred = all_lane_pred[layer_idx]

            glane_pred = glane_pred.view(
                glane_pred.shape[0],
                self.num_group,
                self.num_query,
                glane_pred.shape[-1])
            gcls_pred = gcls_pred.view(
                gcls_pred.shape[0],
                self.num_group,
                self.num_query,
                gcls_pred.shape[-1])

            per_xs_loss = 0.0
            per_zs_loss = 0.0
            per_vis_loss = 0.0
            per_cls_loss = 0.0
            batch_size = len(matched_indices[0])

            for b_idx in range(len(matched_indices[0])):
                for group_idx in range(self.num_group):
                    pred_idx = matched_indices[group_idx][b_idx][0]
                    gt_idx = matched_indices[group_idx][b_idx][1]

                    cls_pred = gcls_pred[:, group_idx, ...]
                    lane_pred = glane_pred[:, group_idx, ...]

                    if gt_idx.shape[0] < 1:
                        cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                        cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)
                        per_cls_loss = per_cls_loss + cls_loss
                        # Fake loss for unused parameters bug
                        per_xs_loss = per_xs_loss + 0.0 * lane_pred[b_idx].mean()
                        continue

                    pos_lane_pred = lane_pred[b_idx][pred_idx]
                    gt_lane = gt_lanes[b_idx][gt_idx]

                    pred_xs = pos_lane_pred[:, :self.code_size]
                    pred_zs = pos_lane_pred[:, self.code_size : 2*self.code_size]
                    pred_vis = pos_lane_pred[:, 2*self.code_size:]
                    gt_xs = gt_lane[:, :self.code_size]
                    gt_zs = gt_lane[:, self.code_size : 2*self.code_size]
                    gt_vis = gt_lane[:, 2*self.code_size:3*self.code_size]

                    loc_mask = gt_vis > 0
                    xs_loss = self.reg_crit(pred_xs, gt_xs)
                    zs_loss = self.reg_crit(pred_zs, gt_zs)
                    xs_loss = (xs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), 1)
                    zs_loss = (zs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), 1)
                    vis_loss = self.bce_loss(pred_vis, gt_vis)

                    cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                    cls_target[pred_idx] = torch.argmax(
                        gt_lane[:, 3*self.code_size:], dim=1)
                    cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)

                    per_xs_loss += xs_loss
                    per_zs_loss += zs_loss
                    per_vis_loss += vis_loss
                    per_cls_loss += cls_loss

            return tuple(map(lambda x: x / batch_size / self.num_group,
                             [per_xs_loss, per_zs_loss, per_vis_loss, per_cls_loss]))

        all_xs_loss, all_zs_loss, all_vis_loss, all_cls_loss = multi_apply(
            single_layer_loss, range(all_lane_pred.shape[0]))
        all_xs_loss = sum(all_xs_loss) / num_layers
        all_zs_loss = sum(all_zs_loss) / num_layers
        all_vis_loss = sum(all_vis_loss) / num_layers
        all_cls_loss = sum(all_cls_loss) / num_layers

        return dict(
            all_xs_loss=self.xs_loss_weight * all_xs_loss,
            all_zs_loss=self.zs_loss_weight * all_zs_loss,
            all_vis_loss=self.vis_loss_weight * all_vis_loss,
            all_cls_loss=self.cls_loss_weight * all_cls_loss,
        )

    def get_3d_keypos(self, 
                      points, 
                      T_lidar2img, 
                      img_shape, 
                      pad_shape,
                      img_feats,
                      point_feats,
                      hit_mask=None,
                      pt2img_feats=None,
                      mf_pt2img_feats=None,
                      mf_hit_mask=None,
                      x=None,
                      **kwargs):
        top_view_region = self.top_view_region
        xmin, ymin, zmin, xmax, ymax, zmax = top_view_region
        output_dict = {}
        points_xyz = torch.cat([
            points[..., :3],
            torch.ones([*points.shape[:-1], 1],
                        dtype=points.dtype,
                        device=points.device)
        ], dim=-1)
        coords_img = ground2img(
            points_xyz, *img_shape,
            T_lidar2img, pad_shape,
            extra_feats=None if self.num_lidar_feat == 3 else points[..., 3:self.num_lidar_feat]
        )
        ground_coords = coords_img[:, :3, ...]
        ground_coords[:, 0, ...] = (ground_coords[:, 0, ...] - xmin) / (xmax - xmin)
        ground_coords[:, 1, ...] = (ground_coords[:, 1, ...] - ymin) / (ymax - ymin)
        ground_coords[:, 2, ...] = (ground_coords[:, 2, ...] - zmin) / (zmax - zmin)

        ground_coords = torch.cat([
            ground_coords, coords_img[:, 4:, ...]], dim=1)

        key_pos = self.pos_encoding_3d(ground_coords)

        output_dict.update(dict(key_pos=key_pos))
        return output_dict