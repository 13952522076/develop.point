import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch
from torch import einsum
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from typing import Any
from pointnet2_ops.pointnet2_utils import furthest_point_sample


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        """
        Implement self-attention layer
        :param dim: input data dim
        :param heads: attention heads
        :param dim_head: dimension in each head
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5  # the 1/sqrt(d_k) in Eq.1 in Attention all you need
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x):
        """
        :input x: [b batch, p points, k nerigbhors, d dimension]
        :return: [b batch, p points, k nerigbhors, d dimension]
        """
        b, k, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b k n (h d) -> b k h n d', h=h), qkv)
        dots = einsum('b k h i d, b k h j d -> b k h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b k h i j, b k h j d -> b k h i d', attn, v)
        out = rearrange(out, 'b k h n d -> b k n (h d)')
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, **kwargs):
        """
        Building Transformer block
        :param dim: input data dimension
        :param heads: heads number
        :param dim_head: dimension in each head
        :param kwargs:
        """
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = Attention(dim=dim, heads=heads, dim_head=dim_head)
        self.ffn = FeedForward(dim=dim, hidden_dim=dim)  # modify hidden_dim accordingly, e.g, 2*dim or dim/2.

    def forward(self, x):
        """
        :input x: [b batch, p points, k nerigbhors, d dimension]
        :return: [b batch, p points, k nerigbhors, d dimension]
        """
        att = self.attention(self.norm1(x))
        att = att + x
        out = self.ffn(self.norm2(att))
        out = out + att
        return out


class TransformerDown(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=0, **kwargs):
        """
        linearly gather neigbors to sampled points by points + offsets by attentional weights
        :param in_dim: input data dimension
        :param out_dim: output data dimension
        :param hid_dim: projection dimension for k, q, if 0, no projection
        :param kwargs:
        """
        super(TransformerDown, self).__init__()
        # self.k = nn.Linear(in_dim, hid_dim) if hid_dim != 0 else nn.Identity()
        # self.q = nn.Linear(in_dim, hid_dim) if hid_dim != 0 else nn.Identity()
        # self.scale = (hid_dim ** -0.5) if hid_dim != 0 else (in_dim ** -0.5)
        # self.v = nn.Linear(in_dim, out_dim)
        self.m = nn.Linear(in_dim, out_dim)

    def forward(self, x, y):
        """
        :input x: farthest points sampling [b batch, p points, 1, d dimension]
        :input y: corresponding neighbors  [b batch, p points, k neigbors, d dimension]
        :return: [b batch, p points, k nerigbhors, d dimension]
        """
        # x: farthest points sampling   [b, p, 1, d]
        # y: corresponding neighbors    [b, p, k, d]
        # return gather data [b, p, 1, out_dim]
        out = self.m(x)
        # q = self.q(x)
        # k = self.k(y)
        # v = self.v(y)
        # dots = einsum('b p k d, b p i d -> b p k i', k, q) * self.scale  # i=1 actually
        # weights = dots.softmax(dim=-2)
        # offset = (weights * v).sum(dim=-2, keepdim=True)
        # out = out + offset
        out = F.relu(out, inplace=True)
        return out


class FPSKNNGrouper(nn.Module):
    def __init__(self, points, knn=16, **kwargs):
        """
        Given a list of unordered data, return the fps neighbors (first neighbor is the sampled point).
        :param points: number of sampled data points
        :param knn: k-neighbors for each sampled point
        :param kwargs:
        """
        super(FPSKNNGrouper, self).__init__()
        self.points = points  # points number of Farthest Points Sampling
        self.knn = knn  # number of k neighbors

    def forward(self, x):
        """
        :param x: input data points corrdications [b, n, 3+c] first 3 dims are coordinates
        :return: grouped_points [b,points, knn, 3+c]
        !!! Notice that: the sampled points = grouped_points[:,:,0,:]
        """
        # sampeld_points = index_points(x, farthest_point_sample(x[:, :, :3], self.points))  # [b,points, 3]
        idx = furthest_point_sample((x[:, :, :3]).contiguous(), self.points).long()
        sampeld_points = index_points(x, idx)  # [b,points, 3]
        distances = square_distance(sampeld_points[:, :, :3], x[:, :, :3])  # including sampled points self.
        knn_idx = distances.argsort()[:, :, :self.knn]
        grouped_points = index_points(x, knn_idx)  # [b,points, knn, 3+c]
        return grouped_points


class Simpler2(nn.Module):
    def __init__(self, num_classes=40, use_normals=False, points=512,
                 blocks=[1, 2, 1, 1], embed_channel=32, k_neighbors=[16, 16, 16, 16],
                 heads=8, dim_head=16, expansion=2, reducer=4, pool="avg", **kwargs):
        super(Simpler2, self).__init__()
        self.stages = len(blocks)
        self.num_classes = num_classes
        channel = 6 if use_normals else 3
        self.linear = nn.Linear(channel, embed_channel)
        self.transformer_stages = nn.ModuleList()
        self.transformer_downs = nn.ModuleList()
        self.groupers = nn.ModuleList()
        for stage, block_num in enumerate(blocks):
            # for appending transformer blocks
            factor = expansion ** stage
            factor_d = int(math.sqrt(factor))
            factor_h = factor // factor_d
            transformer_blocks = []
            for _ in range(block_num):
                transformer_blocks.append(
                    TransformerBlock(dim=embed_channel * factor, heads=heads * factor_h, dim_head=dim_head * factor_d)
                )
            transformer_blocks = nn.Sequential(*transformer_blocks)
            self.transformer_stages.append(transformer_blocks)

            # for appending transformer groups
            knn = k_neighbors[stage]
            self.groupers.append(FPSKNNGrouper(points=points // (reducer ** stage), knn=knn))

            # for appending transformer downs
            self.transformer_downs.append(
                TransformerDown(in_dim=embed_channel * factor, out_dim=embed_channel * factor * expansion,
                                hid_dim=embed_channel)
            )

        self.pool = nn.AdaptiveAvgPool1d(1) if pool=="avg" else nn.AdaptiveMaxPool1d(1)
        self.classify = nn.Linear(embed_channel * factor * expansion, num_classes)


    def forward(self, x):
        x = x.transpose(1,2)
        # x shape: [b, n, d]
        coords = x[:,:,:3]
        out = self.linear(x)
        for i in range(self.stages):
            out = torch.cat([coords, out], dim=-1)
            out = self.groupers[i](out)  # [b,p,k,3+c]
            coords, features = out[:,:,:,:3], out[:,:,:, 3:]
            features = self.transformer_stages[i](features)
            sampled_points = (features[:, :, 0, :]).unsqueeze(dim=-2)
            coords = coords[:,:,0,:]
            out = self.transformer_downs[i](sampled_points, features).squeeze(dim=-2)

        # now, out shape is [b, sampled points, d]
        out = self.pool(out.transpose(1,2)).squeeze(dim=-1)
        out = self.classify(out)
        return {
            "logits": out
        }


def Simpler2A(num_classes=40, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, blocks=[1, 1, 1, 1], reducer=4, **kwargs)

def Simpler2Amax(num_classes=40, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, blocks=[1, 1, 1, 1], reducer=4, pool="max", **kwargs)

def Simpler2B(num_classes=40, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, blocks=[1, 1, 1, 1], reducer=4, dim_head=32, **kwargs)

def Simpler2C(num_classes=40, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, blocks=[1, 1, 1, 1], reducer=4, heads=16, **kwargs)

def Simpler2D(num_classes=40, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, blocks=[2, 2, 2, 2], reducer=4, **kwargs)

def Simpler2E(num_classes=40, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, blocks=[3, 4, 6, 3], reducer=4, **kwargs)

def Simpler2Emax(num_classes=40, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, blocks=[3, 4, 6, 3], reducer=4, pool="max", **kwargs)


def Simpler2Fmax(num_classes=40, use_normals=True, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, use_normals=use_normals,
                    points=512, blocks=[1, 1, 1, 1], embed_channel=32,
                    k_neighbors=[32, 16, 16, 8], heads=8, dim_head=16,
                    expansion=2, reducer=4, pool="max", **kwargs)

def Simpler2Gmax(num_classes=40, use_normals=True, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, use_normals=use_normals,
                    points=512, blocks=[1, 1, 1, 1], embed_channel=64,
                    k_neighbors=[32, 16, 16, 8], heads=8, dim_head=16,
                    expansion=2, reducer=4, pool="max", **kwargs)

def Simpler2Imax(num_classes=40, use_normals=True, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, use_normals=use_normals,
                    points=512, blocks=[1, 1, 1, 1], embed_channel=64,
                    k_neighbors=[16, 16, 16, 16], heads=8, dim_head=32,
                    expansion=2, reducer=4, pool="max", **kwargs)

def Simpler2Hmax(num_classes=40, use_normals=True, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, use_normals=use_normals,
                    points=512, blocks=[2, 2, 2, 2], embed_channel=64,
                    k_neighbors=[32, 16, 16, 8], heads=8, dim_head=16,
                    expansion=2, reducer=4, pool="max", **kwargs)



def Simpler2Jmax(num_classes=40, use_normals=True, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, use_normals=use_normals,
                    points=512, blocks=[1, 1, 1, 1], embed_channel=64,
                    k_neighbors=[16, 16, 16, 16], heads=8, dim_head=16,
                    expansion=2, reducer=4, pool="max", **kwargs)

def Simpler2Kmax(num_classes=40, use_normals=True, **kwargs: Any) -> Simpler2:
    return Simpler2(num_classes=num_classes, use_normals=use_normals,
                    points=512, blocks=[1, 1, 1, 1], embed_channel=64,
                    k_neighbors=[16, 16, 16, 16], heads=8, dim_head=32,
                    expansion=2, reducer=4, pool="max", **kwargs)

if __name__ == '__main__':
    print("===> testing Simpler2Fmax ...")
    pointsformer = Simpler2Fmax()
    data = torch.rand(2, 6, 1024)
    out = pointsformer(data)
    print(out["logits"].shape)


    print("===> testing attention module ...")
    data = torch.rand(32, 64, 6, 128)  # [b batch, p points, k nerigbhors, d dimension]
    model = Attention(128)
    out = model(data)
    print(out.shape)

    print("===> testing TransformerBlock module ...")
    data = torch.rand(32, 64, 6, 128)  # [b batch, p points, k nerigbhors, d dimension]
    model = TransformerBlock(128)
    out = model(data)
    print(out.shape)

    print("===> testing TransformerDown module ...")
    x = torch.rand(32, 64, 1, 128)  # [b batch, p points, k nerigbhors, d dimension]
    y = torch.rand(32, 64, 16, 128)  # [b batch, p points, k nerigbhors, d dimension]
    model = TransformerDown(in_dim=128, out_dim=256, hid_dim=0)
    out = model(x, y)
    print(out.shape)

    print("===> testing farthest_point_sample function random or consistent ...")
    data = torch.rand(10, 64, 3)
    out1 = farthest_point_sample(data, 5)
    out2 = farthest_point_sample(data, 5)
    print(out1 == out2)

    print("===> testing FPSKNNGrouper function ...")
    b, n, points, knn = 4, 256, 128, 16
    data = torch.rand(b, n, 3)
    grouper = FPSKNNGrouper(points=points, knn=knn)
    grouped = grouper(data)
    print(grouped.shape)

    print("===> testing Simpler2A ...")
    pointsformer = Simpler2A()
    data = torch.rand(2, 6, 1024)
    out = pointsformer(data)
    print(out["logits"].shape)

    print("===> testing Simpler2Amax ...")
    pointsformer = Simpler2Amax()
    data = torch.rand(2, 6, 1024)
    out = pointsformer(data)
    print(out["logits"].shape)

    print("===> testing Simpler2B ...")
    pointsformer = Simpler2B()
    data = torch.rand(2, 6, 1024)
    out = pointsformer(data)
    print(out["logits"].shape)

    print("===> testing Simpler2C ...")
    pointsformer = Simpler2C()
    data = torch.rand(2, 6, 1024)
    out = pointsformer(data)
    print(out["logits"].shape)

    print("===> testing Simpler2D ...")
    pointsformer = Simpler2D()
    data = torch.rand(2, 6, 1024)
    out = pointsformer(data)
    print(out["logits"].shape)

    print("===> testing Simpler2E ...")
    pointsformer = Simpler2E()
    data = torch.rand(2, 6, 1024)
    out = pointsformer(data)
    print(out["logits"].shape)



