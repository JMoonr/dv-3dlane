import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange, Reduce
from torch.nn.init import xavier_uniform_, kaiming_normal_, xavier_normal_

from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding


class QPositionEncoding(object):
    LiDAR_XYZ_PE = 0
    LiDAR_Ray_PE = 1


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    # posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class PositionEmbeddingSine(nn.Module):
    def __init__(self, embed_dims, num_pt_feats, num_pos_feats=128, temperature=10000, normalize=False):
        super().__init__()
        self.num_pt_feats = num_pt_feats
        self.num_pos_feats = num_pos_feats
        self.temp = temperature
        self.normalize = normalize
        in_c = embed_dims*3//2
        if num_pt_feats > 3:
            in_c += (num_pt_feats - 3)
        self.q_embedding = nn.Sequential(
            nn.Linear(in_c, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
    
    def forward(self, xyz_hw, input_hw=True, with_pt_feats=True):
        """
        xyz_hw : B x 3 [or 5] x h x w
        return : B x hw x C
        """
        if input_hw:
            xyz_hw = xyz_hw.flatten(-2, -1).permute(0, 2, 1).contiguous()
        pos_emb_3d_sine = pos2posemb3d(xyz_hw[..., :3])
        
        if xyz_hw.shape[1] > 3 and with_pt_feats:
            pos_emb_3d_sine = torch.cat(
                [pos_emb_3d_sine, xyz_hw[..., 3:self.num_pt_feats]], dim=-1)
        return self.q_embedding(pos_emb_3d_sine)


class PositionEmbeddingMLP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.pos_enc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        """
        x      : B x C_in x h x w
        return : B x hw x C_out
        """
        return self.pos_enc(x).flatten(2, 3).permute(0, 2, 1).contiguous()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, in_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [mid_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim]))
            # nn.Conv2d(n, k, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSineMLP(nn.Module):
    def __init__(self, out_with_ln=True, **sinemlp_cfg):
        super().__init__()
        self.sine = build_positional_encoding(sinemlp_cfg['sine_cfg'])
        self.mlp = MLP(**sinemlp_cfg['mlp_cfg'])
        if out_with_ln:
            self.ln = nn.LayerNorm(sinemlp_cfg['mlp_cfg']['out_dim'])
        else:
            self.ln = None
        
    def forward(self, x, mask=None):
        pe = self.sine(x)
        pe = self.mlp(torch.einsum('nchw->nhwc', pe).contiguous())
        if self.ln is None:
            return pe.flatten(1, 2).contiguous()
        return self.ln(pe).flatten(1, 2).contiguous()
    

class LatrMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, with_normalinit=False):
        super().__init__()
        # print('this ')
        self.mlp = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0),
            # Rearrange('b c h w -> b (h w) c')
            )
        if with_normalinit:
            self.init_weights()

    
    def init_weights(self):
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                if i == 0:
                    kaiming_normal_(m, mode='fan_in', nonlinearity='relu')
                else:
                    xavier_normal_(m.weight, gain=1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                xavier_normal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
    def forward(self, x, input_n_m=True, **kwargs):
        # import pdb;pdb.set_trace()
        # if input_n_m:
        permute_input_shape = 'b n m 1 p -> b (1 p) n m'
        permute_output_shape = 'b (1 p) n m -> b n m 1 p'
        # else:
        #     permute_input_shape = 'b nm p -> b p 1 nm'
        #     permute_output_shape = 'b p 1 nm -> b nm (p 1)'
        return einops.rearrange(
            self.mlp(einops.rearrange(x, permute_input_shape)), permute_output_shape)


class BEVFormerPE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=None, out_dim=256, **kwargs):
        super().__init__()
        print('that')
        self.embed_dims = hidden_dim or out_dim
        self.position_encoder = nn.Sequential(
            nn.Linear(input_dim, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, out_dim), 
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, refpts_xyz, pt_first_out=False, **kwargs):
        """
        refpts_xyz : B x N x 3
        """
        # if pt_first_out:
        #     return einops.rearrange(
        #         self.position_encoder(refpts_xyz),
        #         'b n m 1 c -> (n m) b (1 c)')
        return self.position_encoder(refpts_xyz)
        

pos_encoding_map = dict(
    Sine3D = PositionEmbeddingSine,
    MLP = PositionEmbeddingMLP,
    BEVSineMLP = PositionEmbeddingSineMLP,
    sine = PositionEmbeddingSine,
    latrmlp = LatrMLP,
    bevformer_mlp = BEVFormerPE,
)


def build_pos_encoding_layer(pos_encoding_cfg):
    if pos_encoding_cfg['type'] == 'SinePositionalEncoding':
        print(f' >>> use mmcv Encode Pos')
        return build_positional_encoding(pos_encoding_cfg)
    else:
        pos_encod_type = pos_encoding_cfg.pop('type')
        pos_encod_mtd = pos_encoding_map[pos_encod_type]
        print(f' >>> use {pos_encod_type} Encode Pos')
        return pos_encod_mtd(**pos_encoding_cfg)
