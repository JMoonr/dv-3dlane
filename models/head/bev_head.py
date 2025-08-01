import torch
from torch import nn
import torch.nn.functional as F

from mmseg.core import build_pixel_sampler
from mmseg.models.builder import build_loss
from mmseg.ops import resize
from mmseg.models.losses import accuracy
from mmdet3d.models.builder import HEADS
from models.model_init import xavier_init


class SimpleBEVSegHead(nn.Module):
    def __init__(self, 
                 in_channels,
                 num_classes,
                 seg_bev_loss_weight=1.0,
                 loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                 ignore_index=255,
                 mask_seg_loss=False,
                 sampler=None,
                 align_corners=False
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # self.seg_bev_loss_weight = seg_bev_loss_weight
        outc = in_channels
        
        self.seg_bev_pred_layer = nn.Sequential(
            nn.Conv2d(outc, outc // 2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(outc // 2),
            nn.ReLU(True),
            nn.Conv2d(outc // 2, outc // 4, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(outc // 4),
            nn.ReLU(True),
            nn.Conv2d(outc // 4, num_classes, 1, bias=True),
        )

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
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")
    
    def loss(self, seg_logits, seg_label, mask=None):
        # bev_losses:
        """Compute segmentation loss."""
        
        if seg_label.ndim == 4:
            seg_label = torch.max(seg_label, dim=1)[1]
        if self.num_classes == 2:
            seg_label = (seg_label > 0).long()

        loss = dict()

        seg_label = seg_label.unsqueeze(1)
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

    def forward(self, point_feats):
        if isinstance(point_feats, (list, tuple)):
            point_feats = point_feats[0]
        seg_bev_pred = self.seg_bev_pred_layer(point_feats)
        
        return seg_bev_pred

