# define functions to generate cls & reg prediction head
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob



def build_cls_reg_heads(reg_fcs, embed_dims, num_classes, reg_out_c, 
                        num_pred, share_pred_heads=True):
    cls_branch = ClsHead(embed_dims, num_classes=num_classes, 
                         num_reg_fcs=reg_fcs,)
    reg_branch = RegHead(embed_dims, num_classes=num_classes,
                         num_reg_fcs=reg_fcs, reg_dim=reg_out_c
    )

    cls_branches = nn.ModuleList(
        [cls_branch if share_pred_heads else copy.deepcopy(cls_branch) for _ in range(num_pred)])
    reg_branches = nn.ModuleList(
        [reg_branch if share_pred_heads else copy.deepcopy(reg_branch) for _ in range(num_pred)])

    return cls_branches, reg_branches


def build_refpts_pred_layer(embed_dims, num_pt_coords=2):
    return  nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(True),
            nn.Linear(embed_dims, num_pt_coords))


def build_fc_layer(in_dim, mid_dim, out_dim, activation='relu', inplace=False):
    return nn.Sequential(
                nn.Linear(in_dim, mid_dim),
                nn.ReLU(True) if inplace else nn.ReLU(),
                nn.Linear(mid_dim, out_dim),
            )


class GFlatPredLayer(nn.Module):
    def __init__(self, embed_dims, gflat_dim=2):
        super(GFlatPredLayer, self).__init__()
        self.embed_dims = embed_dims
        self.gflat_pred = nn.Sequential(
                nn.Conv2d(embed_dims + 4, embed_dims, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(True),
                nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dims, embed_dims, 1),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(True),
                nn.Conv2d(embed_dims, embed_dims // 4, 1),
                nn.BatchNorm2d(embed_dims // 4),
                nn.ReLU(True),
                nn.Conv2d(embed_dims // 4, gflat_dim, 1))

    def forward(self, x):
        return self.gflat_pred(x)


class ClsHead(nn.Module):
    def __init__(self, embed_dims, num_classes, num_reg_fcs=1, loss_use_sigmoid=True):
        super(ClsHead, self).__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_reg_fcs = num_reg_fcs
        self.loss_use_sigmoid = loss_use_sigmoid
        
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)
        self._init_weights()

    def _init_weights(self):
        if self.loss_use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.xavier_uniform_(self.cls_branch[-1].weight)
            nn.init.constant_(self.cls_branch[-1].bias, 0)

    def forward(self, x):
        return self.cls_branch(x)


class RegHead(nn.Module):
    def __init__(self, embed_dims, num_classes, num_reg_fcs=1, 
                 reg_dim=3):
        super(RegHead, self).__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_reg_fcs = num_reg_fcs
        
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        
        reg_branch.append(
            nn.Linear(self.embed_dims, reg_dim))
        
        reg_branch = nn.Sequential(*reg_branch)
        
        self.reg_branch = reg_branch
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.reg_branch[-1].weight)
        nn.init.constant_(self.reg_branch[-1].bias.data, 0)
    
    def forward(self, x):
        return self.reg_branch(x)

