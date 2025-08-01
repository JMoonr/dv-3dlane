import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from scipy.ndimage import gaussian_filter
import numpy as np
import math

from mmcv.cnn import bias_init_with_prob
from .scatter_utils import scatter_mean, scatter_min, scatter


def norm_ysteps(anchor_y_steps, pc_range):
    y_min, y_max = pc_range[1], pc_range[4]
    return (anchor_y_steps - y_min) / (y_max - y_min)


class GaussianBlur2d(nn.Module):
    def __init__(self, channel, kernel_size, sigma=3):
        super().__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1
        self.sigma = sigma
        self.seq = nn.Sequential(
            nn.Conv2d(
                channel, channel,
                kernel_size, stride=1,
                padding=kernel_size // 2,
                bias=False, groups=channel)
        )
        self.weights_init()

    @torch.no_grad()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((self.kernel_size, self.kernel_size))
        n[self.kernel_size // 2, self.kernel_size // 2] = 1
        k = gaussian_filter(n, sigma=self.sigma)
        k = k / k.max()
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad_(False)


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class Morphology(nn.Module):
    '''
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure. 
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=False)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    @torch.no_grad()
    def forward(self, x):
        assert False
        '''
        x: tensor of shape (B,C,H,W)
        '''
        h, w = x.shape[-2:]

        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        # L = x.size(-1)
        # L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1) # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError
        
        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False) # (B, Cout, L)
        else:
            x = torch.logsumexp(x*self.beta, dim=2, keepdim=False) / self.beta # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, h, w)  # (B, Cout, L/2, L/2)

        return x 


class Dilation2d(Morphology):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')

class Erosion2d(Morphology):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')


def get_downsampled_lidar_depth(lidar_depth, downsample_factor, dep_bound_max):
        batch_size, num_sweeps, num_cams, height, width = lidar_depth.shape
        lidar_depth = lidar_depth.view(
            batch_size * num_sweeps * num_cams,
            height // downsample_factor,
            downsample_factor,
            width // downsample_factor,
            downsample_factor,
            1,
        )
        lidar_depth = lidar_depth.permute(0, 1, 3, 5, 2, 4).contiguous()
        lidar_depth = lidar_depth.view(
            -1, downsample_factor * downsample_factor)# -1 x 16*16
        gt_depths_tmp = torch.where(lidar_depth == 0.0, lidar_depth.max(),
                                    lidar_depth)
        lidar_depth = torch.min(gt_depths_tmp, dim=-1).values
        lidar_depth = lidar_depth.view(batch_size, num_sweeps, num_cams, 1,
                                       height // downsample_factor,
                                       width // downsample_factor)
        lidar_depth = lidar_depth / dep_bound_max
        return lidar_depth


def ground2img(coords3d, H, W, lidar2img, ori_shape,
               mask=None, return_img_pts=False, extra_feats=None,
               with_clone=True, nearest_min=False, reduce='mean'):
    if with_clone:
        coords3d = coords3d.clone()

    if coords3d.ndim == 4:
        coords3d = coords3d.flatten(1, 2)
        if extra_feats is not None:
            extra_feats = extra_feats.flatten(1, 2)
    img_pt = coords3d @ lidar2img.permute(0, 2, 1)
    # img_pt[..., :2] = img_pt[..., :2] / torch.clamp(img_pt[..., 2:3], 1e-5)
    # img_pt[..., :2] = img_pt[..., :2] / torch.maximum(
    #     img_pt[..., 2:3], torch.ones_like(img_pt[..., 2:3]) * 1e-5)
    img_pt = torch.cat([
        img_pt[..., :2] / torch.maximum(
            img_pt[..., 2:3], torch.ones_like(img_pt[..., 2:3]) * 1e-5),
        img_pt[..., 2:]
    ], dim=-1)
    img_pt = img_pt.contiguous()

    # For debug
    if False:
        canvas_np = np.zeros(
            (int(ori_shape[0][0].item()),
             int(ori_shape[0][1].item())))
        img_pt_np = img_pt[0].detach().cpu().numpy()
        for pt in img_pt_np:
            x = int(pt[0])
            y = int(pt[1])
            if x > 0 and x < canvas_np.shape[1] and \
                    y > 0 and y < canvas_np.shape[0]:
                canvas_np[y, x] = 1
        canvas_np = (255 * canvas_np).astype(np.uint8)
        cv2.imwrite('./debug/canvas_np.png', canvas_np)
        import pdb; pdb.set_trace()
    # if input_batch:
    org_h, org_w = ori_shape[0][0], ori_shape[0][1]
    # else:
    #     org_h, org_w = ori_shape[0], ori_shape[1]
    x = img_pt[..., 0] / org_w * (W - 1)
    y = img_pt[..., 1] / org_h * (H - 1)
    valid = (x >= 0) * (y >= 0) * (x <= (W - 1)) \
          * (y <= (H - 1)) * (img_pt[..., 2] > 0)
    if return_img_pts:
        return x, y, valid

    if mask is not None:
        valid = valid * mask.flatten(1, 2).float()

    # B, C, H, W = img_feats.shape
    B = coords3d.shape[0]
    canvas = torch.zeros(
        (B, H, W, 3 + 1 + (extra_feats.shape[-1] if extra_feats is not None else 0)),
        dtype=torch.float32,
        device=coords3d.device)

    x = x.long()
    y = y.long()
    # B x N
    ind = (x + y * W) * valid.long()
    # ind = torch.clamp(ind, 0, H * W - 1)
    ind = ind.long().unsqueeze(-1).repeat(1, 1, canvas.shape[-1])
    canvas = canvas.flatten(1, 2)
    target = coords3d.clone()
    if extra_feats is not None:
        target = torch.cat([target, extra_feats], dim=-1)

    scatter(target, ind, out=canvas, dim=1, reduce=reduce)
    
    canvas = canvas.view(B, H, W, canvas.shape[-1]
        ).permute(0, 3, 1, 2).contiguous()
    
    canvas[:, :, 0, 0] = 0
    # For debug
    if False:
        mask = canvas[0, -1, ...]
        mask = mask.detach().cpu().numpy()
        mask = (mask > 0) * 255
        mask = mask.astype(np.uint8)
        cv2.imwrite('./debug/mask.png', mask)
        # import pdb; pdb.set_trace()
    return canvas


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class ClassBranch(nn.Module):
    def __init__(self, 
                 num_fcs, 
                 in_channels, 
                 mid_channels, 
                 out_channels,
                 loss_use_sigmoid=True):
        super().__init__()
        self.loss_use_sigmoid=loss_use_sigmoid
        cls_branch = []
        for _ in range(num_fcs):
            cls_branch.append(nn.Linear(in_channels, mid_channels))
            cls_branch.append(nn.LayerNorm(mid_channels))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(mid_channels, out_channels))
        self.fc_cls = nn.Sequential(*cls_branch)
        self._init_weights()

    def _init_weights(self):
        if self.loss_use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.xavier_uniform_(self.fc_cls[-1].weight)
            nn.init.constant_(self.fc_cls[-1].bias, 0)
    
    def forward(self, x):
        return self.fc_cls(x)


class RegBranch(nn.Module):
    def __init__(self, num_fcs, in_channels, mid_channels, out_channels):
        super().__init__()
        reg_branch = []
        for _ in range(num_fcs):
            reg_branch.append(nn.Linear(in_channels, mid_channels))
            reg_branch.append(nn.ReLU())
            reg_branch.append(nn.Linear(mid_channels, out_channels))
        self.reg_branch = nn.Sequential(*reg_branch)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.reg_branch[-1].weight)
        nn.init.constant_(self.reg_branch[-1].bias.data, 0)

    def forward(self, x):
        return self.reg_branch(x)


def build_head(head_cfg):
    pass


class SE(nn.Module):
    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)
        self._init_weights()
    
    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
                # init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                # if m.bias is not None:
                #     m.bias.data.zero_()

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x*torch.sigmoid(out)


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "LN": lambda channels: nn.LayerNorm(channels),
        }[norm]
    return norm(out_channels)



class SE1d(nn.Module):
    def __init__(self, in_chnls, ratio, ndim_index=0):
        super(SE1d, self).__init__()
        # self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Linear(in_chnls, in_chnls // ratio)
        self.excitation = nn.Linear(in_chnls // ratio, in_chnls)
        self.ndim_index = ndim_index

    def forward(self, x):
        out = torch.mean(x, dim=self.ndim_index, keepdim=True)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x * torch.sigmoid(out)

class deepFeatureExtractor_EfficientNet(nn.Module):
    def __init__(self, architecture="EfficientNet-B5", lv6=False, lv5=False, lv4=False, lv3=False):
        super(deepFeatureExtractor_EfficientNet, self).__init__()
        assert architecture in ["EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", 
                                    "EfficientNet-B4", "EfficientNet-B5", "EfficientNet-B6", "EfficientNet-B7"]
        
        if architecture == "EfficientNet-B0":
            self.encoder = geffnet.tf_efficientnet_b0_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B1":
            self.encoder = geffnet.tf_efficientnet_b1_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B2":
            self.encoder = geffnet.tf_efficientnet_b2_ns(pretrained=True)
            self.dimList = [16, 24, 48, 120, 1408] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 48, 120, 352] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B3":
            self.encoder = geffnet.tf_efficientnet_b3_ns(pretrained=True)
            self.dimList = [24, 32, 48, 136, 1536] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 48, 136, 384] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B4":
            self.encoder = geffnet.tf_efficientnet_b4_ns(pretrained=True)
            self.dimList = [24, 32, 56, 160, 1792] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 56, 160, 448] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B5":
            self.encoder = geffnet.tf_efficientnet_b5_ns(pretrained=True)
            self.dimList = [24, 40, 64, 176, 2048] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 40, 64, 176, 512] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B6":
            self.encoder = geffnet.tf_efficientnet_b6_ns(pretrained=True)
            self.dimList = [32, 40, 72, 200, 2304] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 40, 72, 200, 576] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B7":
            self.encoder = geffnet.tf_efficientnet_b7_ns(pretrained=True)
            self.dimList = [32, 48, 80, 224, 2560] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 48, 80, 224, 640] #5th feature is extracted after blocks[6]
        del self.encoder.global_pool
        del self.encoder.classifier
        #self.block_idx = [3, 4, 5, 7, 9] #5th feature is extracted after blocks[6]
        #self.block_idx = [3, 4, 5, 7, 10] #5th feature is extracted after conv_head
        self.block_idx = [3, 4, 5, 7, 11] #5th feature is extracted after bn2
        if lv6 is False:
            del self.encoder.blocks[6]
            del self.encoder.conv_head
            del self.encoder.bn2
            del self.encoder.act2
            self.block_idx = self.block_idx[:4]
            self.dimList = self.dimList[:4]
        if lv5 is False:
            del self.encoder.blocks[5]
            self.block_idx = self.block_idx[:3]
            self.dimList = self.dimList[:3]
        if lv4 is False:
            del self.encoder.blocks[4]
            self.block_idx = self.block_idx[:2]
            self.dimList = self.dimList[:2]
        if lv3 is False:
            del self.encoder.blocks[3]
            self.block_idx = self.block_idx[:1]
            self.dimList = self.dimList[:1]
        # after passing blocks[3]    : H/2  x W/2
        # after passing blocks[4]    : H/4  x W/4
        # after passing blocks[5]    : H/8  x W/8
        # after passing blocks[7]    : H/16 x W/16
        # after passing conv_stem    : H/32 x W/32
        self.fixList = ['blocks.0.0','bn']

        for name, parameters in self.encoder.named_parameters():
            if name == 'conv_stem.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        block_cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'act2':
                break
            if k == 'blocks':
                for m, n in v._modules.items():
                    feature = n(feature)
                    try:
                        if self.block_idx[block_cnt] == cnt:
                            out_featList.append(feature)
                            block_cnt += 1
                            break
                        cnt += 1
                    except:
                        continue
            else:
                feature = v(feature)
                if self.block_idx[block_cnt] == cnt:
                    out_featList.append(feature)
                    block_cnt += 1
                    break
                cnt += 1            
            
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
