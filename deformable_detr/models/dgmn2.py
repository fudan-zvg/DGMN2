import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_

from util.misc import NestedTensor
from .dgmn2_block import DGMN2Block


__all__ = [
    'dgmn2_tiny',
    'dgmn2_small',
    'dgmn2_medium',
    'dgmn2_large'
]


class PatchEmbed_stage1(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=768, mid_embed_dim=384):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, mid_embed_dim, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.norm1 = nn.BatchNorm2d(mid_embed_dim)
        self.conv2 = nn.Conv2d(mid_embed_dim, mid_embed_dim, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.norm2 = nn.BatchNorm2d(mid_embed_dim)
        self.conv3 = nn.Conv2d(mid_embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm(self.conv(x)))

        return x


class DGMN2(nn.Module):
    def __init__(
        self, in_chans=3, embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), qkv_bias=False,
        qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=(3, 4, 6, 3)
    ):
        super().__init__()
        self.depths = depths
        self.strides = [8, 16, 32]
        self.num_channels = embed_dims[1:]

        # patch_embed
        self.patch_embed1 = PatchEmbed_stage1(in_chans=in_chans, embed_dim=embed_dims[0], mid_embed_dim=embed_dims[0] // 2)
        self.patch_embed2 = PatchEmbed(in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([DGMN2Block(
            fea_size=(224 // 4, 224 // 4), dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer)
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([DGMN2Block(
            fea_size=(224 // 8, 224 // 8), dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer)
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([DGMN2Block(
            fea_size=(224 // 16, 224 // 16), dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer)
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([DGMN2Block(
            fea_size=(224 // 32, 224 // 32), dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer)
            for i in range(depths[3])])

        # init weights
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward_features(self, x_nested):
        out: Dict[str, NestedTensor] = {}
        x = x_nested.tensors

        # stage 1
        x = self.patch_embed1(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # stage 2
        x = self.patch_embed2(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        m = x_nested.mask
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out["0"] = NestedTensor(x, mask)

        # stage 3
        x = self.patch_embed3(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        m = x_nested.mask
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out["1"] = NestedTensor(x, mask)

        # stage 4
        x = self.patch_embed4(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        for blk in self.block4:
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        m = x_nested.mask
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out["2"] = NestedTensor(x, mask)

        return out

    def forward(self, x):
        x = self.forward_features(x)

        return x


class dgmn2_tiny(DGMN2):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(8, 8, 4, 4), qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=(2, 2, 2, 2)
        )


class dgmn2_small(DGMN2):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(8, 8, 4, 4), qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=(3, 4, 6, 3)
        )


class dgmn2_medium(DGMN2):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(8, 8, 4, 4), qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=(3, 4, 18, 3)
        )


class dgmn2_large(DGMN2):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(8, 8, 4, 4), qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=(3, 8, 27, 3)
        )
