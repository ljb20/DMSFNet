import torch.nn as nn
from model.vision_transformer import Attention
from model.vision_transformer import DropPath
from model.vision_transformer import Mlp
from model.vision_transformer import trunc_normal_


class ReUnit(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ReBlock(nn.Module):

    def __init__(self, dim, num_heads,depth=1, mode=0):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.mode = mode
        for i in range(self.depth):
            self.blocks.append(
                ReUnit(dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                       attn_drop=0.,
                       drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Reconstruct(nn.Module):

    def __init__(self, dim, num_heads,depth=1):
        super().__init__()
        self.re1 = ReBlock(dim, num_heads, depth=depth)
        
    def forward(self, x):
        re1 = self.re1(x)
        return re1


class CRM(nn.Module):

    def __init__(self, dim, num_heads, depth=1, miss=None):
        super().__init__()
        self.RGBRE = Reconstruct(dim, num_heads, depth=depth)
        self.NIRE = Reconstruct(dim, num_heads, depth=depth)
        print("CRM HERE!!!")
        self.miss = miss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, ma, mb, mc=None):
        if self.training:
            RGB_NI = self.RGBRE(ma)
            NI_RGB = self.NIRE(mb)
            loss_rgb = nn.MSELoss()(RGB_NI, mb)
            loss_ni = nn.MSELoss()(NI_RGB, ma)
            loss = loss_rgb + loss_ni
            return loss
        else:
            if self.miss == None:
                pass
            