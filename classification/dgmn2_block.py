import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from dcn import DeformUnfold


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RelPosEmb(nn.Module):
    def __init__(self, fmap_size, dim_head, num_samples):
        super().__init__()
        height, width = fmap_size
        scale = dim_head ** -0.5
        self.num_samples = num_samples
        self.rel_height = nn.Parameter(torch.randn(height + num_samples - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width + num_samples - 1, dim_head) * scale)

    def rel_to_abs(self, x):
        b, h, l, c = x.shape
        x = torch.cat((x, torch.zeros((b, h, l, 1), dtype=x.dtype, device=x.device)), dim=3)
        x = x.reshape(b, h, l * (c + 1))
        x = torch.cat((x, torch.zeros((b, h, self.num_samples - 1), dtype=x.dtype, device=x.device)), dim=2)
        x = x.reshape(b, h, l + 1, self.num_samples + l - 1)
        x = x[:, :, :l, (l - 1):]
        return x

    def relative_logits_1d(self, q, rel_k):
        logits = torch.matmul(q, rel_k.transpose(0, 1))
        b, h, x, y, r = logits.shape
        logits = logits.reshape(b, h * x, y, r)
        logits = self.rel_to_abs(logits)
        return logits

    def forward(self, q):
        rel_logits_w = self.relative_logits_1d(q, self.rel_width)

        q = q.transpose(2, 3)
        rel_logits_h = self.relative_logits_1d(q, self.rel_height)

        return rel_logits_w + rel_logits_h


class DGMN2Attention(nn.Module):
    def __init__(self, dim, num_heads=8, fea_size=(224, 224), qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # sample
        self.num_samples = 9
        self.conv_offset = nn.Linear(self.head_dim, self.num_samples * 2, bias=qkv_bias)
        self.unfold = DeformUnfold(kernel_size=3, padding=1, dilation=1)

        # relative position
        self.pos_emb = RelPosEmb(fea_size, self.head_dim, self.num_samples)

    def forward(self, x, H, W):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        offset = self.conv_offset(x.reshape(B, N, self.num_heads, self.head_dim)).permute(0, 2, 3, 1).reshape(B * self.num_heads, self.num_samples * 2, H, W)

        k = k.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, H, W)
        v = v.transpose(2, 3).reshape(B * self.num_heads, self.head_dim, H, W)
        k = self.unfold(k, offset).transpose(1, 2).reshape(B, self.num_heads, N, self.head_dim, self.num_samples)
        v = self.unfold(v, offset).reshape(B, self.num_heads, self.head_dim, self.num_samples, N).permute(0, 1, 4, 3, 2)

        attn = torch.matmul(q.unsqueeze(3), k) * self.scale

        attn_pos = self.pos_emb(q.reshape(B * self.num_heads, 1, H, W, self.head_dim)).reshape(B, self.num_heads, N, 1, self.num_samples)
        attn = attn + attn_pos

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DGMN2Block(nn.Module):
    def __init__(
        self, dim, num_heads, fea_size=(224, 224), mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DGMN2Attention(
            dim, num_heads=num_heads, fea_size=fea_size, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
