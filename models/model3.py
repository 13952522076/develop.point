"""
Based on model2, using the Transformer block.
The extraction is doubled for depth.

Learning Point Cloud with Progressively Local representation.
[B,3,N] - {[B,G,K,d]-[B,G,d]}  - {[B,G',K,d]-[B,G',d]} -cls
"""
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
import torch.nn.functional as F
from _utils import square_distance, index_points, farthest_point_sample, query_ball_point, knn_point
from pointnet2_ops import pointnet2_utils


class LocalGrouper(nn.Module):
    def __init__(self, groups, kneighbors, **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,2d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long() # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        # grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        grouped_points = index_points(points, idx)
        grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
        new_points = torch.cat([grouped_points_norm,
                                new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)]
                               , dim=-1)
        return new_xyz, new_points


class FCBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(FCBNReLU1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class FCBNReLU1DRes(nn.Module):
    def __init__(self, channel, kernel_size=1, bias=False):
        super(FCBNReLU1DRes, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(channel)
        )

    def forward(self, x):
        identity = x
        return F.relu(self.net(x)+identity, inplace=True)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        """
        input and output: [b batch, p points,  d dimension]
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

    def forward(self, x):
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
        [b batch, p points,  d dimension]
        :param dim: input data dimension
        :param heads: heads number
        :param dim_head: dimension in each head
        :param kwargs:
        """
        super(TransformerBlock, self).__init__()
        self.attention = Attention(dim=dim, heads=heads, dim_head=dim_head)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim,1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        """
        :input x: [b batch, p points,  d dimension]
        :return: [b batch, p points, d dimension]
        """
        features = x
        att = self.attention(features)
        att = att + features
        out = self.ffn(att.permute(0,2,1))
        out = out.permute(0,2,1) + att
        return out




class PreExtraction(nn.Module):
    def __init__(self, channels, blocks=1, heads=8, dim_head=64):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                TransformerBlock(channels, heads, dim_head)  # [b,p,d]
            )
        self.operation = nn.Sequential(*operation)
    def forward(self, x):
        b, g, k, d = x.size()  # [b,g,k,d]
        x = x.reshape(-1, k, d)    # [b*g, k, d]
        batch_size, _, N = x.size()
        x = self.operation(x)  # [b*g, k, d]
        x = x.permute(0,2,1)  # [b*g, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  #[b*g, d]
        x = x.reshape(b, g, -1)  #[b, g, d]
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, heads=8, dim_head=64):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                TransformerBlock(channels, heads, dim_head)  # [b,p,d]
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        return self.operation(x)


class Model3(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, heads=4,
                 pre_blocks=[2,2,2,2], pos_blocks=[2,2,2,2], k_neighbors=[32,32,32,32],
                 reducers=[2,2,2,2], **kwargs):
        super(Model3, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.heads = heads
        self.points=points
        self.embedding = nn.Sequential(
            FCBNReLU1D(3, embed_dim),
            FCBNReLU1D(embed_dim, embed_dim)
        )
        assert len(pre_blocks)==len(k_neighbors)==len(reducers)==len(pos_blocks), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel*2
            pre_block_num=pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points//reduce

            dim_head = out_channel*2//self.heads

            # append local_grouper_list
            local_grouper = LocalGrouper(anchor_points, kneighbor) #[b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(out_channel, pre_block_num, heads=self.heads, dim_head=dim_head)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, heads=self.heads, dim_head=dim_head)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.classifier = nn.Sequential(
            nn.Linear(last_channel, last_channel//4),
            nn.BatchNorm1d(last_channel//4),
            nn.ReLU(),
            nn.Linear(last_channel//4, self.class_num)
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x) # B,D,N
        x= x.permute(0, 2, 1)
        for i in range(self.stages):
            # xyz: [b,p,3]->[b,g,3]; x (input is tansported): [b,p,d]->[b,g,k,d]
            xyz, x = self.local_grouper_list[i](xyz, x)
            x = self.pre_blocks_list[i](x)  #[b, g, k, d] -> [b, g, d]
            x = self.pos_blocks_list[i](x)  #[b,p,d] -> [b,p,d]

        x = x.permute(0, 2, 1)  #[b,d,p]
        x = F.adaptive_max_pool1d(x,1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


def model3A(num_classes=40, **kwargs) -> Model3:
    return Model3(points=1024, class_num=num_classes, embed_dim=64,
                 pre_blocks=[1,1,1,1], pos_blocks=[1,1,1,1], k_neighbors=[32,32,32,32],
                 reducers=[2,2,2,2], **kwargs)

def model3B(num_classes=40, **kwargs) -> Model3:
    return Model3(points=1024, class_num=num_classes, embed_dim=64,
                 pre_blocks=[1,1,1], pos_blocks=[1,1,1], k_neighbors=[32,32,32],
                 reducers=[2,2,2], **kwargs)


def model3C(num_classes=40, **kwargs) -> Model3:
    return Model3(points=1024, class_num=num_classes, embed_dim=32,
                 pre_blocks=[1,1,1,1], pos_blocks=[1,1,1,1], k_neighbors=[32,32,32,32],
                 reducers=[2,2,2,2], **kwargs)


def model3D(num_classes=40, **kwargs) -> Model3:
    return Model3(points=1024, class_num=num_classes, embed_dim=64,
                 pre_blocks=[2,2,2,2], pos_blocks=[2,2,2,2], k_neighbors=[32,32,32,32],
                 reducers=[2,2,2,2], **kwargs)


def model3E(num_classes=40, **kwargs) -> Model3:
    return Model3(points=1024, class_num=num_classes, embed_dim=64,
                 pre_blocks=[2,2,2], pos_blocks=[2,2,2], k_neighbors=[32,32,32],
                 reducers=[2,2,2], **kwargs)

def model3F(num_classes=40, **kwargs) -> Model3:
    return Model3(points=1024, class_num=num_classes, embed_dim=32,
                 pre_blocks=[2,2,2], pos_blocks=[2,2,2], k_neighbors=[32,32,32],
                 reducers=[2,2,2], **kwargs)

if __name__ == '__main__':
    batch, groups,neighbors,dim=2,512,32,16
    x = torch.rand(batch,groups,neighbors,dim)
    pre_extractor = PreExtraction(dim,3)
    out = pre_extractor(x)
    print(out.shape)

    x = torch.rand(batch, groups, dim)
    pos_extractor = PosExtraction(dim, 3)
    out = pos_extractor(x)
    print(out.shape)


    data = torch.rand(2, 3, 1024)
    print("===> testing model ...")
    model = Model3()
    out = model(data)
    print(out.shape)

    print("===> testing model3A ...")
    model = model3A()
    out = model(data)
    print(out.shape)

    print("===> testing modelB ...")
    model = model3B()
    out = model(data)
    print(out.shape)

    print("===> testing modelC ...")
    model = model3C()
    out = model(data)
    print(out.shape)

    print("===> testing modelD ...")
    model = model3D()
    out = model(data)
    print(out.shape)

    print("===> testing modelE ...")
    model = model3E()
    out = model(data)
    print(out.shape)
