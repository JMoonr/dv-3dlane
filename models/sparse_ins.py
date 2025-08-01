import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from mmcv.cnn import ConvModule

from .sparse_inst_loss import SparseInstCriterion, SparseInstMatcher
from .utils import SE
from .norm_utils import build_iam_norm_layer


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)


class MaskBranch(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.hidden_dim
        num_convs = cfg.num_convs
        kernel_dim = cfg.kernel_dim
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


class InstanceBranch(nn.Module):
    def __init__(self, cfg, in_channels, **kwargs):
        super().__init__()
        num_mask = cfg.num_query
        dim = cfg.hidden_dim
        num_classes = cfg.num_classes
        kernel_dim = cfg.kernel_dim
        num_convs = cfg.num_convs
        num_group = cfg.get('num_group', 1)
        sparse_num_group = cfg.get('sparse_num_group', 1)
        self.num_group = num_group
        self.sparse_num_group = sparse_num_group
        self.num_mask = num_mask
        self.inst_convs = _make_stack_3x3_convs(
                            num_convs=num_convs, 
                            in_channels=in_channels, 
                            out_channels=dim)

        self.iam_conv = nn.Conv2d(
            dim * num_group,
            num_group * num_mask * sparse_num_group,
            3, padding=1, groups=num_group * sparse_num_group)
        self.fc = nn.Linear(dim * sparse_num_group, dim)
        # output
        self.mask_kernel = nn.Linear(
            dim, kernel_dim)
        self.cls_score = nn.Linear(
            dim, num_classes)
        self.objectness = nn.Linear(
            dim, 1)
        self.prior_prob = 0.01

        self.iam_norm_layer = build_iam_norm_layer(
            cfg.get('iam_norm_layer', dict(type='iam')))
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        c2_xavier_fill(self.fc)

    def forward(self, seg_features, is_training=True):
        out = {}
        # SparseInst part
        seg_features = self.inst_convs(seg_features)
        # predict instance activation maps
        iam = self.iam_conv(seg_features.tile(
            (1, self.num_group, 1, 1)))
        if not is_training:
            iam = iam.view(
                iam.shape[0],
                self.num_group,
                self.num_mask * self.sparse_num_group,
                *iam.shape[-2:])
            iam = iam[:, 0, ...]
            num_group = 1
        else:
            num_group = self.num_group

        B, N = iam.shape[:2]
        C = seg_features.size(1)

        iam_prob_norm_hw = self.iam_norm_layer(iam)

        # aggregate features: BxCxHxW -> Bx(HW)xC
        all_inst_features = torch.bmm(
            iam_prob_norm_hw,
            seg_features.view(B, C, -1).permute(0, 2, 1).contiguous()) #BxNxC

        # concat sparse group features
        # all_inst_feat: B x group<grouo-detr, here 1> x sparse_group<4> x query_num x C<256>
        # -> B x group x query_num x (sparse_group x C)
        inst_features1 = all_inst_features.reshape(
            B, num_group,
            self.sparse_num_group,
            self.num_mask, -1
        ).permute(0, 1, 3, 2, 4).reshape(
            B, num_group,
            self.num_mask, -1).contiguous()
        inst_features = F.relu(
            self.fc(inst_features1))

        # avg over sparse group
        
        # iam_prob = iam_prob.view(
        #     B, num_group,
        #     self.sparse_num_group,
        #     self.num_mask,
        #     iam_prob.shape[-1])
        # iam_prob = iam_prob.mean(dim=2).flatten(1, 2)

        inst_features = inst_features.flatten(1, 2)
        # iam_prob_norm_hw = iam_prob_norm_hw.view(B, num_group, self.sparse_num_group, self.num_mask, -1)
        # iam_prob_norm_hw = iam_prob_norm_hw.mean(dim=2).flatten(1,2)#.view(B, self.num_mask, *iam.shape[-2:]).contiguous()
        # import pdb; pdb.set_trace()

        out.update(dict(
            iam=iam,
            iam_prob_norm_hw=iam_prob_norm_hw,
            inst_features=inst_features.clone()))
        if is_training:
            pred_logits = self.cls_score(inst_features)
            pred_kernel = self.mask_kernel(inst_features)
            pred_scores = self.objectness(inst_features)
            out.update(dict(
                pred_logits=pred_logits,
                pred_kernel=pred_kernel,
                pred_scores=pred_scores))
        return out
        # return inst_features, pred_logits, \
        #         pred_kernel, pred_scores, \
        #         iam, iam_prob, 

class GroupInstanceBranch(nn.Module):
    def __init__(self, cfg, in_channels, **kwargs) -> None:
        super().__init__()
        num_mask = cfg.num_query
        dim = cfg.hidden_dim
        num_classes = cfg.num_classes
        kernel_dim = cfg.kernel_dim
        num_convs = cfg.num_convs
        num_group = cfg.get('num_group', 1)
        sparse_num_group = cfg.get('sparse_num_group', 1)
        # group detr
        self.num_group = num_group
        # org sparse_ins group
        self.sparse_num_group = sparse_num_group
        self.num_mask = num_mask
        self.inst_convs = _make_stack_3x3_convs(
                            num_convs=num_convs, 
                            in_channels=in_channels, 
                            out_channels=dim)

        self.iam_conv = nn.Conv2d(
            dim * num_group,
            num_group * num_mask * sparse_num_group,
            3, padding=1, groups=num_group * sparse_num_group)
        
        expand_dim = dim * sparse_num_group
        self.fc = nn.Linear(expand_dim, expand_dim)
        # output
        self.mask_kernel = nn.Linear(
            expand_dim, kernel_dim)
        self.cls_score = nn.Linear(
            expand_dim, num_classes)
        self.objectness = nn.Linear(
            expand_dim, 1)
        self.prior_prob = 0.01
        
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        c2_xavier_fill(self.fc)

    def forward(self, seg_features, is_training=True):
        out = {}
        # SparseInst part
        seg_features = self.inst_convs(seg_features)
        # predict instance activation maps
        iam = self.iam_conv(seg_features.tile(
            (1, self.num_group, 1, 1)))
        if not is_training:
            iam = iam.view(
                iam.shape[0],
                self.num_group,
                self.num_mask * self.sparse_num_group,
                *iam.shape[-2:])
            iam = iam[:, 0, ...]
            num_group = 1
        else:
            num_group = self.num_group

        iam_prob = iam.sigmoid()
        B, N = iam_prob.shape[:2]
        C = seg_features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob_norm_hw = iam_prob / normalizer[:, :, None]

        # aggregate features: BxCxHxW -> Bx(HW)xC
        all_inst_features = torch.bmm(
            iam_prob_norm_hw,
            seg_features.view(B, C, -1).permute(0, 2, 1).contiguous()) #BxNxC

        # concat sparse group features
        inst_features = all_inst_features.reshape(
            B, num_group,
            self.sparse_num_group,
            self.num_mask, -1
        ).permute(0, 1, 3, 2, 4).reshape(
            B, num_group,
            self.num_mask, -1).contiguous()
        inst_features = F.relu_(
            self.fc(inst_features))

        # avg over sparse group
        iam_prob = iam_prob.view(
            B, num_group,
            self.sparse_num_group,
            self.num_mask,
            iam_prob.shape[-1])
        iam_prob = iam_prob.mean(dim=2).flatten(1, 2)
        inst_features = inst_features.flatten(1, 2)
        out.update(dict(
            iam_prob=iam_prob,
            inst_features=inst_features))
        if self.training:
            pred_logits = self.cls_score(inst_features)
            pred_kernel = self.mask_kernel(inst_features)
            pred_scores = self.objectness(inst_features)
            out.update(dict(
                pred_logits=pred_logits,
                pred_kernel=pred_kernel,
                pred_scores=pred_scores))
        return out

class SparseInsDecoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        with_pt_center_feats = getattr(cfg.decoder, 'with_pt_center_feats', False)
        self.with_pt_center_feats = with_pt_center_feats
        in_channels = cfg.encoder.out_dims + 2
        if with_pt_center_feats:
            self.fuse_2dfeats_ptimgfeats = SE(
                in_chnls=in_channels + cfg.encoder.out_dims,
                ratio=4)
            self.fuse_2dfeats_ptimgfeats2 = ConvModule(
                in_channels + cfg.encoder.out_dims,
                cfg.encoder.out_dims,
                1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=dict(type='ReLU'),
                inplace=True,
            )
        self.output_iam = cfg.decoder.output_iam
        self.scale_factor = cfg.decoder.scale_factor
        self.sparse_decoder_weight = cfg.sparse_decoder_weight
        self.pe_enc_mtd = getattr(cfg.decoder, 'pe_enc_mtd', None)

        if self.pe_enc_mtd is None:
            pass
        elif self.pe_enc_mtd == 'SE':
            in_channels = in_channels + cfg.encoder.out_dims
            self.pe_encoder = SE(
                                in_chnls=in_channels,
                                ratio=4
                                )
        elif self.pe_enc_mtd == 'cat':
            in_channels = in_channels + cfg.encoder.out_dims
        elif self.pe_enc_mtd == 'sum2feat':
            in_channels = in_channels
        else:
            raise NotImplementedError(
                'Now sparse_deocder: pe_enc_mtd -> SE | cat | sum2feat | None, ONLY. ')

        print(' >>> sparse in_chnls = %s' % in_channels)
        if with_pt_center_feats:
            branch_in_c = cfg.encoder.out_dims
        else:
            branch_in_c = in_channels
        self.inst_branch = InstanceBranch(cfg.decoder, branch_in_c)
        # dim, num_convs, kernel_dim, in_channels
        self.mask_branch = MaskBranch(cfg.decoder, branch_in_c)
        self.sparse_inst_crit = SparseInstCriterion(
            num_classes=cfg.decoder.num_classes,
            matcher=SparseInstMatcher(),
            cfg=cfg)
        self._init_weights()

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)
    
    def _init_weights(self):
        self.inst_branch._init_weights()
        self.mask_branch._init_weights()
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

    def forward(self, features, is_training=True,
                point_feats=None, **kwargs):
        # TODO use point_feats
        output = {}
        coord_features = self.compute_coordinates(features)
        # pos_emb_3d = B x (hw) x C
        B, C, h, w = features.shape

        if self.pe_enc_mtd is not None:
            pos_emb_3d = kwargs['pos_emb_3d'].permute(0, 2, 1).view(B, C, h, w).contiguous()
            if self.pe_enc_mtd == 'sum2feat':
                features = features + pos_emb_3d

        # feat : B x C x h x w
        features = torch.cat([coord_features, features], dim=1)

        if self.pe_enc_mtd is None:
            pass
        elif self.pe_enc_mtd == 'SE':
            features = torch.cat([pos_emb_3d, features], dim=1)
            features = self.pe_encoder(features)
        elif self.pe_enc_mtd == 'cat':
            features = torch.cat([pos_emb_3d, features], dim=1)
        elif self.pe_enc_mtd == 'sum2feat':
            # add before cat 2d_coords, line 354
            features = features
        else:
            raise NotImplementedError('only support: SE | cat')

        # if kwargs.get('pt_center_feats') is not None:
        if self.with_pt_center_feats:
            features = torch.cat([features, kwargs['pt_center_feats']], dim=1)
            features = self.fuse_2dfeats_ptimgfeats2(
                self.fuse_2dfeats_ptimgfeats(features) + features
            )
        
        inst_output = self.inst_branch(features, is_training=is_training)
        output.update(inst_output)

        if is_training:
            mask_features = self.mask_branch(features)
            pred_kernel = inst_output['pred_kernel']
            N = pred_kernel.shape[1]
            B, C, H, W = mask_features.shape

            # if self.training:
            pred_masks = torch.bmm(pred_kernel, mask_features.view(
            B, C, H * W)).view(B, N, H, W)
            pred_masks = F.interpolate(
                pred_masks, scale_factor=self.scale_factor,
                mode='bilinear', align_corners=False)
            output.update(dict(
                pred_masks=pred_masks))

        if is_training: # kd need this
            sparse_inst_losses, matched_indices = self.loss(
                    output,
                    lane_idx_map=kwargs.get('lane_idx_map'),
                    input_shape=kwargs.get('input_shape')
            )
            for k, v in sparse_inst_losses.items():
                sparse_inst_losses[k] = self.sparse_decoder_weight * v
            output.update(sparse_inst_losses)
            output['matched_indices'] = matched_indices
        return output

    def loss(self, output, lane_idx_map, input_shape):
        """
        output : from self.forward
        lane_idx_map : instance-level segmentation map, [20, H, W] where 20=max_lanes
        """
        pred_masks = output['pred_masks']
        pred_masks = output['pred_masks'].view(
            pred_masks.shape[0],
            self.inst_branch.num_group,
            self.inst_branch.num_mask,
            *pred_masks.shape[2:])
        pred_logits = output['pred_logits']
        pred_logits = output['pred_logits'].view(
            pred_logits.shape[0],
            self.inst_branch.num_group,
            self.inst_branch.num_mask,
            *pred_logits.shape[2:])
        pred_scores = output['pred_scores']
        pred_scores = output['pred_scores'].view(
            pred_scores.shape[0],
            self.inst_branch.num_group,
            self.inst_branch.num_mask,
            *pred_scores.shape[2:])

        out = {}
        all_matched_indices = []
        for group_idx in range(self.inst_branch.num_group):
            sparse_inst_losses, matched_indices = \
                self.sparse_inst_crit(
                    outputs=dict(
                        pred_masks=pred_masks[:, group_idx, ...].contiguous(),
                        pred_logits=pred_logits[:, group_idx, ...].contiguous(),
                        pred_scores=pred_scores[:, group_idx, ...].contiguous(),
                    ),
                    targets=self.prepare_targets(lane_idx_map),
                    input_shape=input_shape, # seg_bev
                )
            for k, v in sparse_inst_losses.items():
                out['%s_%d' % (k, group_idx)] = v
            all_matched_indices.append(matched_indices)
        return out, all_matched_indices

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            cls_labels = targets_per_image.flatten(-2).max(-1)[0]
            pos_mask = cls_labels > 0

            target["labels"] = cls_labels[pos_mask].long()
            target["masks"] = targets_per_image[pos_mask] > 0
            new_targets.append(target)
        return new_targets
