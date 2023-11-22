from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, data_size=(1, 40, 40), kernel_size=(4, 10, 10), stride=(1, 1, 1), padding=(2, 5, 5), in_c=12,
                 embed_dim=768, norm_layer=None):
        super().__init__()
        self.data_size = data_size
        self.conv_size = tuple(
            (data_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1 for i in range(len(data_size)))
        self.flatten_size = self.conv_size[0] * self.conv_size[1]  # * self.conv_size[2]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, D, H, W = x.shape
        assert H == self.data_size[0] and W == self.data_size[1], \
            f"Input data size ({D}*{H}*{W}) doesn't match model ({self.data_size[1]}*{self.data_size[2]}*{self.data_size[3]}). "
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class trans_conv(nn.Module):
    def __init__(self, in_c=768, out_c=20, kernel_size=(4, 10, 10), stride=(1, 1, 1), padding=(0, 5, 5),
                 norm_layer=None):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.proj = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride,
                                       padding=padding)
        self.norm = norm_layer(out_c) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 series_len=8,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.series_len = series_len
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):

        B, L, _ = x.shape
        x = x.view(B, self.series_len, L // self.series_len, self.dim)
        B, D, N, C = x.shape
        for j in range(B):
            before_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            for i in range(D):
                xt = x[j, i, :, :].clone()
                qkv = self.qkv(xt)
                qkv = qkv.reshape(1, N, 3, self.num_heads,
                                  torch.div(C, self.num_heads, rounding_mode='trunc'))
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                xt = (attn @ v).transpose(1, 2).reshape(1, N, C)
                xt = self.proj(xt)
                xt = self.proj_drop(xt)
                x[j, i, :, :] = xt.detach().clone()
            after_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_delta = (after_mem_alloc - before_mem_alloc)

        x = x.view(B, L, self.dim)
        return x


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


class Block(nn.Module):
    def __init__(self,
                 dim,
                 series_len,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, series_len=series_len, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Model(nn.Module):
    def __init__(self, data_size=(1, 40, 40), conv_kernel_size=(4, 10, 10),
                 conv_stride=(1, 1, 1),
                 conv_padding=(2, 5, 5),
                 trans_conv_kernel_size=(4, 10, 10),
                 trans_conv_stride=(1, 1, 1),
                 trans_conv_padding=(2, 5, 5), in_c=2, out_c=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 trans_layer=trans_conv, act_layer=None):
        super(Model, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(data_size=data_size, kernel_size=conv_kernel_size, stride=conv_stride,
                                       padding=conv_padding, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.flatten_size

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.s0 = self.patch_embed.conv_size[0]
        self.s1 = self.patch_embed.conv_size[1]
        # self.s2 = self.patch_embed.conv_size[2]

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, series_len=self.s0, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.trans_conv = trans_layer(in_c=embed_dim, out_c=out_c, kernel_size=trans_conv_kernel_size,
                                      stride=trans_conv_stride, padding=trans_conv_padding)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = torch.reshape(x, (x.shape[0], self.s0, self.s1, x.shape[2]))
        x = x.permute(0, 3, 1, 2)
        x = self.trans_conv(x)
        x = torch.squeeze(x)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def model(in_c=5, out_c=35, data_size=(8, 20, 20), conv_kernel_size=(4, 10, 10), conv_stride=(1, 1, 1),
             conv_padding=(2, 0, 0),
             trans_conv_kernel_size=(4, 10, 10),
             trans_conv_stride=(1, 1, 1),
             trans_conv_padding=(2, 0, 0),
             embed_dim=40,
             depth=2,
             num_heads=4):
    model = Model(data_size=data_size,
                              conv_kernel_size=conv_kernel_size,
                              conv_stride=conv_stride,
                              conv_padding=conv_padding,
                              trans_conv_kernel_size=trans_conv_kernel_size,
                              trans_conv_stride=trans_conv_stride,
                              trans_conv_padding=trans_conv_padding,
                              embed_dim=embed_dim,
                              depth=depth,
                              num_heads=num_heads,
                              in_c=in_c,
                              out_c=out_c)
    return model
