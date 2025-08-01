import torch
from torch import nn
from ..registry import READERS

from ..scatter_utils import scatter_mean
from ..ops.pillar_ops.pillar_modules import PillarMaxPooling
    


@READERS.register_module()
class DynamicPFE(nn.Module):
    def __init__(self,
                 in_channels=5,
                 num_channels=(32, ),
                 pillar_size=0.1,
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 ):
        super().__init__()
        if isinstance(pillar_size, (tuple, list)):
            self.pillar_size = pillar_size
        else:
            self.pillar_size = [pillar_size for _ in range(2)]

        self.pc_range = pc_range
        self.in_channels = in_channels
        assert len(num_channels) > 0
        # only use the relative distance to its pillar centers
        num_channels = [2 + in_channels] + list(num_channels)
        self.pfn_layers = PillarMaxPooling(
            mlps=num_channels,
            pillar_size=pillar_size,
            point_cloud_range=pc_range
        )
        self.height, self.width = self.pfn_layers.height, self.pfn_layers.width


    def forward(self, input_pts, **kwargs):
        pts_xy = []
        pts_batch_cnt = []
        pts_features = []
        pts_xyz = []

        canvas = torch.zeros(
            (len(input_pts), self.height, self.width, self.in_channels),
            dtype=torch.float32,
            device=input_pts[0].device)
        
        for i, points in enumerate(input_pts):
            coors_x = ((points[:, 0] - self.pc_range[0]) / self.pillar_size[0]).floor().int()
            coors_y = ((points[:, 1] - self.pc_range[1]) / self.pillar_size[1]).floor().int()
            
            mask = (coors_x >= 0) & (coors_x < self.width) & \
                (coors_y >= 0) & (coors_y < self.height)
            coors_xy = torch.stack((coors_x[mask], coors_y[mask]), dim=1)
            pts_xyz.append(points[mask, :3])

            # # scatter xyz & 2_dim to BEV canvas
            # ind = (coors_x + coors_y * self.width) * mask.long()
            # ind = ind.unsqueeze(-1).repeat(1, canvas[i].shape[-1])
            # canvas_i = canvas[i]
            # canvas_i = canvas_i.flatten(1, 2)
            # target = points.clone()
            # scatter_mean(target, ind, out=canvas_i, dim=1)
            # canvas_i = canvas_i.view(self.height, self.width, canvas_i.shape[-1])
            # canvas_i[0, 0, :] = 0
            # canvas[i] = canvas_i
    
            pts_xy.append(coors_xy)
            pts_features.append(points[mask])
            pts_batch_cnt.append(len(coors_xy))

        pts_xy = torch.cat(pts_xy)
        pts_batch_cnt = pts_xy.new_tensor(pts_batch_cnt, dtype=torch.int32)
        pts_features = torch.cat(pts_features)
        sparse_tensor, pillars_xyz = self.pfn_layers(
            pts_xy, pts_batch_cnt, pts_features, pts_xyz)

        # canvas = canvas.permute(0, 3, 1, 2).contiguous()
        return sparse_tensor, pillars_xyz
