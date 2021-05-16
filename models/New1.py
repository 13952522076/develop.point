"""
Including:
    1. class token, k neighbors + 1, in each transformer block stage, as develop8, Done.
    2. group conv (in_dim, head_dim*heads, groups=heads), replac FC(in_dim, head_dim*heads) maps, as develop9, Done.
    3. positional encoding in Trans block, fea + encoded, as develop6max, Done.
    4. Different from develop4Amax, we add two droput(0.1) to FFN. Done
"""
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
        # project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5  # the 1/sqrt(d_k) in Eq.1 in Attention all you need
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Conv1d(dim, inner_dim, 1, groups=heads, bias=False)
        self.to_k = nn.Conv1d(dim, inner_dim, 1, groups=heads, bias=False)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, groups=heads, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim,1,bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)

        )

        # self.position = nn.Sequential(
        #     nn.Linear(3, dim // 8),
        #     nn.LayerNorm(dim // 8),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim // 8, dim),
        #     nn.LayerNorm(dim),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        """
        :input x: [b batch, p points,  d dimension]
        # :input coords: [b batch, p points, 3]
        :return: [b batch, p points,  d dimension]
        """
        trans = x.permute(0,2,1)
        b, d, p, h = *trans.shape, self.heads
        query = self.to_q(trans)
        key = self.to_k(trans)
        value = self.to_v(trans)
        query, key, value = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=h), [query, key, value])
        dots = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, value)
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out).permute(0,2,1)
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
        self.attention = Attention(dim=dim, heads=heads, dim_head=dim_head)
        # self.ffn = FeedForward(dim=dim, hidden_dim=dim)  # modify hidden_dim accordingly, e.g, 2*dim or dim/2.
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim,1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        """
        :input x: [b batch, p points,  3+d dimension]
        :return: [b batch, p points, 3+d dimension]
        """
        coords, features = x[:, :, :3], x[:, :, 3:]
        # relative_position = coords.unsqueeze(dim=2) - coords.unsqueeze(dim=1)
        # pos = self.position(relative_position)  #[n,p,p,dim]
        att = self.attention(features)
        att = att + features
        out = self.ffn(att.permute(0,2,1))
        out = out.permute(0,2,1) + att
        out = torch.cat([coords, out], dim=-1)
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


class LocalGather(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        """
        :input: [b,p,k,3+c]
        :output: [b,p,3+c]
        :param channel:
        :param kwargs:
        """
        super(LocalGather, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):  # x [b,p,k,3+c], p: selected points, k is nerighbor number
        xyz = (x[:,:,0,:3]).contiguous()
        fea = (x[:,:,:,3:]).contiguous()
        # off_set = fea - (fea[:,:,0,:]).unsqueeze(dim=-2)
        # fea = torch.cat([fea, off_set], dim=-1)
        fea = fea.permute(0,3,1,2)
        fea = self.fcn(fea)
        # fea = (fea.max(dim=-1)[1]).permute(0,2,1)
        # return torch.cat([xyz,fea],dim=-1)
        fea = (fea[:,:,:,0]).permute(0,2,1)
        return torch.cat([xyz, fea], dim=-1)


class New1(nn.Module):
    def __init__(self, num_classes=40, use_normals=False, points=512,
                 blocks=[1, 2, 1, 1], embed_channel=32, k_neighbors=[16, 16, 16, 16],
                 heads=8, dim_head=16, expansion=2, reducer=4, pool="max", **kwargs):
        super(New1, self).__init__()
        self.stages = len(blocks)
        self.num_classes = num_classes
        channel = 6 if use_normals else 3
        self.embed = nn.Sequential(
            nn.Conv1d(channel, embed_channel,1),
            nn.BatchNorm1d(embed_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_channel, embed_channel,1),
            nn.BatchNorm1d(embed_channel),
            nn.ReLU(inplace=True)
        )
        self.local_gathers = nn.ModuleList()
        self.transformer_stages = nn.ModuleList()
        # self.class_tokens = nn.ParameterList()
        self.groupers = nn.ModuleList()
        for stage, block_num in enumerate(blocks):
            # for appending transformer blocks
            factor = expansion ** stage
            factor_d = int(math.sqrt(factor))
            factor_h = factor // factor_d
            transformer_blocks = []
            for _ in range(block_num):
                transformer_blocks.append(
                    TransformerBlock(dim=embed_channel * factor * expansion, heads=heads * factor_h, dim_head=dim_head * factor_d)
                )
            transformer_blocks = nn.Sequential(*transformer_blocks)
            self.transformer_stages.append(transformer_blocks)

            # # for class token
            # self.class_tokens.append( nn.Parameter(torch.rand(1, 1, 1, embed_channel * factor + 3)) )

            # for appending transformer groups
            knn = k_neighbors[stage]
            self.groupers.append(FPSKNNGrouper(points=points // (reducer ** stage), knn=knn))

            # for appending local gathers
            self.local_gathers.append(
                LocalGather(in_channel=embed_channel*factor, out_channel=embed_channel * factor * expansion)
            )
            # # for appending transformer downs
            # self.transformer_downs.append(
            #     TransformerDown(in_dim=embed_channel * factor, out_dim=embed_channel * factor * expansion,
            #                     hid_dim=embed_channel)
            # )

        self.pool = nn.AdaptiveAvgPool1d(1) if pool=="avg" else nn.AdaptiveMaxPool1d(1)
        # self.classify = nn.Linear(embed_channel * factor * expansion, num_classes)
        self.classify = nn.Sequential(
            nn.Linear(embed_channel * factor * expansion, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)

        )


    def forward(self, x):  # x[b,d,n]
        coords = (x[:,:3,:]).contiguous()  # [b,3,n]
        out = self.embed(x)  # [b,embed_dim, n]

        coords = coords.permute(0,2,1)  # [b,n,3]
        out = out.permute(0,2,1)  # [b,n,embed_dim]
        out = torch.cat([coords, out], dim=-1)
        for i in range(self.stages):
            out = self.groupers[i](out)  # [b,p,k,3+c], p: selected points, k is nerighbor number
            out = self.local_gathers[i](out)  # [b, p, 3+c]
            out = self.transformer_stages[i](out)
            # coords, features = out[:, :, :3], out[:, :, 3:]
            # # print(f"features.shape: {features.shape}")
            # sampled_points = (features[:, :, 0, :]).unsqueeze(dim=-2)
            # coords = coords[:, :, 0, :]
            # out = self.transformer_downs[i](sampled_points)

        # now, out shape is [b, sampled points, d]
        xyz, fea = out[:,:,:3], out[:,:,3:]
        out = self.pool(fea.transpose(1,2)).squeeze(dim=-1)
        out = self.classify(out)
        return out


def new1A(num_classes=40, **kwargs: Any) -> New1:
    return New1(num_classes=num_classes, blocks=[2, 2, 2, 2], reducer=4, k_neighbors=[32,32,32,32], **kwargs)

def new1B(num_classes=40, **kwargs: Any) -> New1:
    return New1(num_classes=num_classes, blocks=[1, 1, 1, 1], reducer=4, k_neighbors=[32,32,32,32], **kwargs)

def new1C(num_classes=40, **kwargs: Any) -> New1:
    return New1(num_classes=num_classes, blocks=[1, 1, 1, 1], reducer=4, k_neighbors=[16,16,16,16], **kwargs)


def new1D(num_classes=40, **kwargs: Any) -> New1:
    return New1(num_classes=num_classes, blocks=[2, 3, 6, 4], reducer=4, k_neighbors=[32,32,32,32], **kwargs)

if __name__ == '__main__':
    print("===> testing localgather...")
    channel=16
    gather = LocalGather(in_channel=channel, out_channel=2*channel)
    data = torch.rand(2,512,32,channel+3)
    out = gather(data)
    print(out.shape)

    data = torch.rand(2, 3, 1024)
    print("===> testing new1A ...")
    pointsformer = new1A()
    out = pointsformer(data)
    print(out.shape)
    print("===> testing new1B ...")
    pointsformer = new1B()
    out = pointsformer(data)
    print(out.shape)

    print("===> testing new1C ...")
    pointsformer = new1C()
    out = pointsformer(data)
    print(out.shape)

    print("===> testing new1D ...")
    pointsformer = new1D()
    out = pointsformer(data)
    print(out.shape)


