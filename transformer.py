import torch.nn as nn
import torch
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
        
        
def fuse_hsp(x, p, group_size=5):
    t = torch.zeros(group_size, x.size(1))
    for i in range(x.size(0)):
        tmp = x[i, :]
        if i == 0:
            nx = tmp.expand_as(t)
        else:
            nx = torch.cat(([nx, tmp.expand_as(t)]), dim=0)
    nx = nx.view(x.size(0) * group_size, x.size(1), 1, 1)
    y = nx.expand_as(p)
    return y


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, group=5, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))
        self.group = group
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))



    def forward(self, x, mask=None):
        identity = x
        bs_gp, dim, wid, hei = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        if bs_gp // self.group == 0:
            bs = 1
            gp = bs_gp
        else:
            bs = bs_gp // self.group
            gp = self.group
        x = x.reshape(bs, gp, dim, wid, hei)
        x = x.permute(0, 1, 3, 4, 2).reshape(bs, gp * wid * hei, dim)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=bs)
        x = torch.cat((cls_tokens, x), dim=1)
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)

        cls_x = x[:, 0, :]
        x = x[:, 1:, :]
        x = x.reshape(bs, gp, wid, hei, dim).permute(0, 1, 4, 2, 3).reshape(bs_gp, dim, wid, hei)

        return x,cls_x
