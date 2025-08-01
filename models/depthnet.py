
import numpy as np
import torch
import torch.nn.functional as F

from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

from .utils import ground2img


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):

    def __init__(self, in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 return_context_feat=False):
        super(DepthNet, self).__init__()
        self.context_channels = context_channels
        self.depth_channels = depth_channels

        self.return_context_feat = return_context_feat

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        mlp_dim = 16
        self.bn = nn.BatchNorm1d(mlp_dim)
        self.depth_mlp = Mlp(mlp_dim, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware

        if return_context_feat:
            self.context_conv = nn.Conv2d(mid_channels,
                                        context_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
            self.context_mlp = Mlp(mlp_dim, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, lidar2img):
        # intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        # batch_size = intrins.shape[0]
        # num_cams = intrins.shape[2]
        # ida = mats_dict['ida_mats'][:, 0:1, ...]
        # sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        # bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
        #                                 4).repeat(1, 1, num_cams, 1, 1)

        # mlp_input = torch.cat(
        #     [
        #         torch.stack(
        #             [
        #                 intrins[:, 0:1, ..., 0, 0],
        #                 intrins[:, 0:1, ..., 1, 1],
        #                 intrins[:, 0:1, ..., 0, 2],
        #                 intrins[:, 0:1, ..., 1, 2],
        #                 ida[:, 0:1, ..., 0, 0],
        #                 ida[:, 0:1, ..., 0, 1],
        #                 ida[:, 0:1, ..., 0, 3],
        #                 ida[:, 0:1, ..., 1, 0],
        #                 ida[:, 0:1, ..., 1, 1],
        #                 ida[:, 0:1, ..., 1, 3],
        #                 bda[:, 0:1, ..., 0, 0],
        #                 bda[:, 0:1, ..., 0, 1],
        #                 bda[:, 0:1, ..., 1, 0],
        #                 bda[:, 0:1, ..., 1, 1],
        #                 bda[:, 0:1, ..., 2, 2],
        #             ],
        #             dim=-1,
        #         ),
        #         sensor2ego.view(batch_size, 1, num_cams, -1),
        #     ],
        #     -1,
        # )
        mlp_input = lidar2img.flatten(1)
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))

        x = self.reduce_conv(x)

        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)

        if self.return_context_feat:
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)
            depth = torch.cat([depth, context], dim=1)

        return depth


class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class DepthNetWrapper(nn.Module):
    def __init__(self, 
                 in_channels=256,
                 mid_channels=256,
                 context_channels=256,
                 depth_channels=128,
                 position_range=None,
                 loss_weight=1.0,
                 with_dyn_cls_weight=None,
                 gen_lidar_depth=True,
                 use_bce_loss=False,
                 depth_resolution=0.4,
                 norm_gt_wo_mask=False,
                 **kwargs,
                 ):
        super().__init__()
        if depth_channels is None:
            depth_channels = int((position_range[4] - position_range[1]) / depth_resolution)
        self.net = DepthNet(
            in_channels=in_channels,
            mid_channels=mid_channels,
            context_channels=context_channels,
            depth_channels=depth_channels,
            **kwargs
        )
        self.depth_channels = depth_channels
        self.position_range = position_range
        self.loss_weight = loss_weight
        self.with_dyn_cls_weight = with_dyn_cls_weight
        self.gen_lidar_depth = gen_lidar_depth
        self.use_bce_loss = use_bce_loss
        self.depth_resolution = depth_resolution
        self.norm_gt_wo_mask = norm_gt_wo_mask
        assert not norm_gt_wo_mask

    def forward(self, img_feats, input_dict, **kwargs):
        if not isinstance(img_feats, (list, tuple)):
            img_feats = [img_feats]
        lidar2img = input_dict['lidar2img']
        depth_pred = self.net(img_feats[0], lidar2img).squeeze(1)

        if self.net.return_context_feat:
            depth_context = depth_pred[:, -self.net.context_channels:]
            depth_pred = depth_pred[:, :self.net.depth_channels]

        out = dict(depth_pred=depth_pred)

        if self.net.return_context_feat:
            out.update(dict(depth_context=depth_context))

        depth_gt, fg_mask = self._gen_depth_label(
            depth_pred.shape[-2:],
            input_dict['points'],
            input_dict['lidar2img'],
            input_dict['pad_shape'],
            depth_gt=input_dict.get('depth_gt', None),
            fg_mask=input_dict.get('fg_mask', None)
        )
        # return for kd (img_branch) usage
        out.update(
                dict(
                    depth_gt = depth_gt,
                    fg_mask = fg_mask
                ))

        if self.training:
            loss = self.loss(
                depth_pred=depth_pred,
                depth_gt=depth_gt,
                fg_mask=fg_mask,
            )
            out.update(dict(depth_loss = loss))

        return out
    
    def _gen_depth_label(self,
                         depth_pred_hw,
                         points,
                         lidar2img,
                         image_shape,
                         depth_gt=None,
                         fg_mask=None):
        """
        1. for mm model, generate depth_gt using LiDAR points
        2. for kd: img model, using depth_gt from mm model generation

        predictions:
            1.new: using BEVDepth pred form: bce
            2.latrmm does: softmax
        """
        if self.gen_lidar_depth:
            depth_gt_xyzm = self.get_lidar_img(
                depth_pred_hw,
                points,
                lidar2img,
                image_shape)
            depth_gt = depth_gt_xyzm[:, 1, ...]
        # else:
        #     # if depth_gt is not None
        #     # for kd usage.
        #     loss = 0.
            
        if self.use_bce_loss:
            if self.gen_lidar_depth:
                fg_mask = depth_gt_xyzm[:, -1, ...]
                depth_gt, fg_mask = self._gen_bce_label(
                    depth_gt, fg_mask)
            else:
                return depth_gt, fg_mask
        else:
            depth_gt, fg_mask = self._gen_old_latrmm_label(depth_gt, depth_gt_xyzm)

        if False:
            plot_random_depth_map(depth_gt, fg_mask)
            # import pdb;pdb.set_trace()
        return depth_gt, fg_mask

    def loss(self,
             depth_pred,
             depth_gt,
             fg_mask=None):

        if self.with_dyn_cls_weight is None:
            cls_weight = None
        elif self.with_dyn_cls_weight == 'log':
            cls_weight = cal_log_disty_weights(
                self.position_range[4],
                self.depth_channels,
                depth_pred.device,
                clip_max=None)
        elif 'log_clip' in self.with_dyn_cls_weight:
            clip_max = float(self.with_dyn_cls_weight.split('log_clip')[-1])
            cls_weight = cal_log_disty_weights(
                self.position_range[4],
                self.depth_channels,
                depth_pred.device,
                clip_max=clip_max)
            # print('depth weight clip max : %s' % clip_max)
        else:
            raise NotImplementedError('ONLY log WEIGHTS support now.')

        if fg_mask is None or (fg_mask.sum() == 0).item():
            return 0

        if self.use_bce_loss:
            loss = F.binary_cross_entropy(
                torch.sigmoid(depth_pred).permute(0, 2, 3, 1)[fg_mask],
                depth_gt[fg_mask],
                reduction='none',
                weight=cls_weight,
            ).sum() / max(1.0, fg_mask.sum())
        else:
            if depth_gt is not None:
                loss = F.cross_entropy(
                    depth_pred,
                    depth_gt,
                    reduction='none',
                    weight=cls_weight
                )
                if mask is not None:
                    loss = (loss * fg_mask).sum() / torch.clamp(loss.sum(), 1)
        
        loss = loss * self.loss_weight
        
        return loss

    def get_lidar_img(self, depth_pred_hw, points,
                      lidar2img, image_shape,
                      reduce='mean'):
        depth_gt_xyzm = ground2img(
            F.pad(points[..., :3], (0, 1), mode='constant', value=1),
            depth_pred_hw[0],
            depth_pred_hw[1],
            lidar2img,
            image_shape,
            reduce=reduce
        )
        # depth_gt = depth_gt_xyzm[:, 1, ...]
        return depth_gt_xyzm

    def _gen_bce_label(self, depth_gt, fg_mask):
        depth_gt = torch.where(
            fg_mask == 0,
            1e5 * torch.ones_like(depth_gt), depth_gt)
        
        depth_gt = (depth_gt -
                    (self.position_range[1] - self.depth_resolution)) / self.depth_resolution
        depth_gt = torch.where(
            (depth_gt < self.depth_channels + 1) & (depth_gt >= 0.0),
            depth_gt, torch.zeros_like(depth_gt))

        depth_gt = F.one_hot(depth_gt.long(), 
                             num_classes=self.depth_channels + 1)[..., 1:].float()
        
        mask = torch.max(depth_gt, dim=-1).values > 0.0
        return depth_gt, mask
    
    def _gen_old_latrmm_label(self, depth_gt, depth_gt_xyzm):
        # old latrmm depth gt process with softmax loss
        depth_gt = (depth_gt - self.position_range[1]) \
                / (self.position_range[4] - self.position_range[1])
        depth_gt = torch.clamp(depth_gt, 0, 1)
        depth_gt = depth_gt * (self.depth_channels - 1)
        depth_gt = depth_gt.long()

        mask = depth_gt_xyzm[:, -1, ...]
        return depth_gt, mask


def build_depth_net(config):
    return DepthNetWrapper(**config)



def cal_log_disty_weights(maxy, cls_num, device, clip_max=None):
    weights = np.arange(cls_num) / (cls_num-1)
    weights = -np.log(weights / (maxy / cls_num * 2)  + 1e-8 )
    weights[0] = weights[3] # to avoid big 0 1 position value
    weights[1] = weights[3]
    if clip_max is not None:
        weights = np.clip(weights, 0.1, clip_max)
    return torch.tensor(weights).float().to(device)


def cal_linear_disty_weights(maxy, cls_num, device):
    weights = torch.arange(cls_num)
    weights =  1 / (weights + 1e-6)
    weights[0] = weights[1]
    weights = torch.FloatTensor(weights, device=device)
    return weights
