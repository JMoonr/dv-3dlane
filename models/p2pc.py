import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import build_loss

from .pos_utils import PositionEmbeddingSine
from .utils import GaussianBlur2d


class P2PC(nn.Module):
    """Point to pixel completion.
    
    point to pixel loss + residual connection of pseudo feat
    
    """
    def __init__(self,
                 img_num_levels=3,
                 img_in_channels=[128, 256, 512],
                 out_channel=256,
                 points_channel=256,
                 num_project_layer=1,
                 fuse_pt_feat=False,
                 loss=dict(type='MSELoss', reduction='none'),
                 loss_weight=1.0,
                 use_fuse_img_feat=False,
                 add_sine3d_pos=False,
                 mask_preprocess=None,
                 mask_sf_and_mf=False,
                 mask_mf=True):
        super().__init__()
        # assert not fuse_pt_feat, 'should have been fused before'
        self.use_fuse_img_feat = use_fuse_img_feat
        self.add_sine3d_pos = add_sine3d_pos
        if add_sine3d_pos:
            self.sine3d_layer = PositionEmbeddingSine(
                embed_dims=out_channel, num_pt_feats=3, temperature=10000
            )
            self.sine3d_layer_fuse_layer = nn.Conv2d(
                out_channel + points_channel,
                out_channel, 1
            )
        if fuse_pt_feat:
            img_in_channels.append(points_channel)
            self.fuse_layers = nn.ModuleList()
            for i, c in enumerate(img_in_channels):
                self.fuse_layers.append(nn.Sequential(
                    nn.Conv2d(c, out_channel, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.Conv2d(out_channel, out_channel, 1)
                ))
        proj_layers = []
        for _ in range(num_project_layer):
            proj_layers.extend(
                [nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
                 nn.BatchNorm2d(out_channel),
                 nn.ReLU()])
        proj_layers.append(nn.Conv2d(out_channel, out_channel, 1))
        self.project_layers = nn.Sequential(*proj_layers)
        self.crit = build_loss(loss)
        self.loss_weight = loss_weight
        self.fuse_pt_feat = fuse_pt_feat
        self.mask_sf_and_mf = mask_sf_and_mf
        self.mask_mf = mask_mf
        self.mask_preprocess = mask_preprocess
        if mask_preprocess is not None:
            processor = []
            for process in mask_preprocess:
                t = process.pop('type')
                if t == 'erode':
                    processor.append(Erosion2d(**process))
                elif t == 'gaussian_blur':
                    processor.append(GaussianBlur2d(**process))
                else:
                    raise NotImplementedError()
            self.mask_preprocess = nn.Sequential(*processor)

    def forward(self, img_feats, hit_mask=None, pt2img_feats=None, fuse_img_feats=None):
        out = {}
        if self.use_fuse_img_feat:
            img_feats = fuse_img_feats
        if self.fuse_pt_feat:
            img_feats = (*img_feats, pt2img_feats)
            fuse_img_feat = self.fuse_mlvl_feats(img_feats)
        else:
            fuse_img_feat = img_feats[0]
        if self.add_sine3d_pos:
            fuse_img_feat = torch.cat(
                [fuse_img_feat,
                 self.sine3d_layer(pt2img_feats[:, :3, ...]).permute(0, 2, 1).view(
                    pt2img_feats.shape[0], -1, *pt2img_feats.shape[-2:]
                 )
                ], dim=1)
            fuse_img_feat = self.sine3d_layer_fuse_layer(fuse_img_feat)
        pseudo_point_feat = self.project_layers(fuse_img_feat)
        return pseudo_point_feat

    def get_loss(self, pseudo_point_feat,
                 hit_mask, pt2img_feats,
                 mf_hit_mask, mf_pt2img_feats):
        p2p_loss = self.crit(pseudo_point_feat, mf_pt2img_feats.detach())
        p2p_loss = p2p_loss.mean(1)

        if self.mask_sf_and_mf:
            mask = mf_hit_mask.float() * (1 - hit_mask.float())
        elif self.mask_mf:
            mask = mf_hit_mask.float()
        else:
            mask = torch.ones_like(p2p_loss)
        if self.mask_preprocess is not None:
            mask = self.mask_preprocess(mask.unsqueeze(1)).squeeze(1)
            mask = mask / torch.clamp(mask.flatten(1).max(1)[0][:, None, None], 1e-6)
        p2p_loss = (p2p_loss * mask).sum() / torch.clamp(mask.sum(), 1)
        p2p_loss * self.loss_weight
        return p2p_loss

    def fuse_mlvl_feats(self, mlvl_feats):
        feats = []
        for i, f in enumerate(mlvl_feats):
            f = self.fuse_layers[i](f)
            if i != 0:
                f = F.interpolate(f, size=mlvl_feats[0].shape[2:])
            feats.append(f)
        return torch.sum(torch.stack(feats, dim=0), dim=0)
