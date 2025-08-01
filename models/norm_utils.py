import torch
import torch.nn as nn
import torch.nn.functional as F



def build_iam_norm_layer(cfg):
    t = cfg.pop('type')
    if t == 'iam':
        return IAM(**cfg)
    elif t == 'softmax':
        return Softmax(**cfg)
    else:
        raise NotImplementedError()


# -------------- keep group ----------------
class IAM(nn.Module):
    def forward(self, iam):
        iam_prob = iam.sigmoid()
        B, N = iam_prob.shape[:2]
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob_norm_hw = iam_prob / normalizer[:, :, None]
        return iam_prob_norm_hw


# -------------- do not keep group --------------
class Softmax(nn.Module):
    def __init__(self, beta=1.0, group_reduce_method='max'):
        super().__init__()
        self.beta = beta
        self.group_reduce_method = group_reduce_method

    def _flat_softmax(self, f):
        _, N, H, W = f.shape
        f = f.reshape(-1, N, H * W)
        h = F.softmax(f, dim=2)
        return h.reshape(-1, N, H, W)

    def forward(self, iam):
        B, num_group, N, h, w = iam.shape
        if self.group_reduce_method == 'max':
            iam = torch.max(iam, dim=1)[0]
        elif self.group_reduce_method == 'mean':
            iam = torch.mean(iam, dim=1)
        else:
            assert False
        iam_prob = self._flat_softmax(iam * self.beta)
        return iam_prob
