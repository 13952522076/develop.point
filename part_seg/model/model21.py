"""
Exactly based on Model10, but ReLU to GeLU
Based on Model8, add dropout and max, avg combine.
Based on Local model, add residual connections.
The extraction is doubled for depth.

Learning Point Cloud with Progressively Local representation.
[B,3,N] - {[B,G,K,d]-[B,G,d]}  - {[B,G',K,d]-[B,G',d]} -cls
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from pointnet2_ops import pointnet2_utils

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
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


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


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


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
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class FCBNReLU1DRes(nn.Module):
    def __init__(self, channel, kernel_size=1, bias=False):
        super(FCBNReLU1DRes, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(channel),
            nn.GELU(),
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(channel)
        )

    def forward(self, x):
        return F.gelu(self.net(x)+x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        # project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim,1),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, **kwargs):
        """
        [b batch, d dimension, k points]
        :param dim: input data dimension
        :param heads: heads number
        :param dim_head: dimension in each head
        :param kwargs:
        """
        super(TransformerBlock, self).__init__()
        self.attention = Attention(dim=dim, heads=heads, dim_head=dim_head)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim)
        )


    def forward(self, x):
        """
        :input x: [b batch, d dimension, p points,]
        :return: [b batch,  d dimension, p points,]
        """
        att = self.attention(x)
        att = F.gelu(att+x)
        out = self.ffn(att)
        out = F.gelu(att+out)
        return out


class PreExtraction(nn.Module):
    def __init__(self, channels, blocks=1):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                FCBNReLU1DRes(channels)
            )
        self.operation = nn.Sequential(*operation)
        self.transformer = TransformerBlock(channels, heads=4)
    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.operation(x)  # [b, d, k]
        x = self.transformer(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                FCBNReLU1DRes(channels)
            )
        self.operation = nn.Sequential(*operation)
        self.transformer = TransformerBlock(channels, heads=4)

    def forward(self, x):  # [b, d, k]
        return self.transformer(self.operation(x))


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1):
        super(FeaturePropagation, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(in_channel, out_channel,1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.GELU()
        )
        layers= []
        for i in range(blocks):
            layers.append(FCBNReLU1DRes(out_channel))
        self.extraction = nn.Sequential(*layers)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points


class Model21(nn.Module):
    def __init__(self, num_part=50, points=2048, embed_dim=128,
                 pre_blocks=[4,4], pos_blocks=[4,4], k_neighbors=[32,32], back_blocks=[2,2],
                 reducers=[4,4], **kwargs):
        super(Model21, self).__init__()
        self.stages = len(pre_blocks)
        self.num_part = num_part
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

        channel_list = [embed_dim]
        for i in range(len(pre_blocks)):
            out_channel = last_channel*2
            pre_block_num=pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points//reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(anchor_points, kneighbor) #[b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(out_channel, pre_block_num)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num)
            self.pos_blocks_list.append(pos_block_module)
            last_channel = out_channel
            channel_list.append(last_channel)

        channel_list.reverse()  # reverse, from last channel to first channel
        self.channel_list = channel_list

        self.cls_label_embedding = nn.Sequential(
            nn.Conv1d(16,64,1,bias=False),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.GELU(),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.GELU()
        )

        self.extra_info = nn.Sequential(
            nn.Conv1d(last_channel+64, 256, 1, bias=False),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.GELU(),
            nn.Conv1d(256, 64, 1, bias=False),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.GELU()
        )

        self.back_block_list = nn.ModuleList()
        last_channel = self.channel_list[0]
        for i in range(len(pre_blocks)):
            self.back_block_list.append(
                FeaturePropagation(
                    last_channel+self.channel_list[i+1]+64, self.channel_list[i], back_blocks[i]
                )
            )
            last_channel = self.channel_list[i]



        self.classifier = nn.Sequential(
            nn.Conv1d(last_channel, last_channel, 1),
            nn.BatchNorm1d(last_channel),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Conv1d(last_channel, last_channel//2, 1),
            nn.BatchNorm1d(last_channel//2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Conv1d(last_channel//2, self.num_part, 1)
        )

    def forward(self, x, norm_plt, cls_label, gt=None):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x) # B,D,N

        fea_list = [x]
        xyz_list = [xyz]

        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))   # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            fea_list.append(x)
            xyz_list.append(xyz)

        fea_list.reverse()  # [b, d, p]
        xyz_list.reverse()  # [b, p, 3]

        global_fea = F.adaptive_max_pool1d(x,1)  # [b, d, 1]
        cls_label = cls_label.view(batch_size, 16, 1)
        cls_label = self.cls_label_embedding(cls_label)
        extra_info = torch.cat([global_fea, cls_label], dim=1)
        extra_info = self.extra_info(extra_info)
        # print(f"extra_info shape: {extra_info.shape}")

        xyz_deeper, fea_deeper = xyz_list[0], fea_list[0]
        for i in range(self.stages):
            xyz_shallower, fea_shallower = xyz_list[i+1], fea_list[i+1]
            _, _, samples = fea_shallower.shape
            fea_shallower = torch.cat([fea_shallower, extra_info.repeat(1,1,samples)],dim=1)
            fea_deeper = self.back_block_list[i](xyz_shallower, xyz_deeper, fea_shallower, fea_deeper)
            xyz_deeper = xyz_shallower

        out = self.classifier(fea_deeper)
        out = F.log_softmax(out, dim=1)
        out = out.permute(0, 2, 1)  # b,n,50

        if gt is not None:
            return out, F.nll_loss(out.contiguous().view(-1, self.num_part), gt.view(-1, 1)[:, 0])
        else:
            return out


def model21H(num_classes=40, **kwargs) -> Model21:
    return Model21(points=1024, class_num=num_classes, embed_dim=128,
                 pre_blocks=[4,4], pos_blocks=[4,4], k_neighbors=[32,32],
                 reducers=[4,4], **kwargs)

if __name__ == '__main__':
    data = torch.rand(2,128,10)
    att = Attention(128)
    out = att(data)
    print(out.shape)



    batch, groups,neighbors,dim=2,512,32,16
    x = torch.rand(batch,groups,neighbors,dim)
    pre_extractor = PreExtraction(dim,3)
    out = pre_extractor(x)
    print(out.shape)

    x = torch.rand(batch, dim, groups)
    pos_extractor = PosExtraction(dim, 3)
    out = pos_extractor(x)
    print(out.shape)


    data = torch.rand(2, 3, 1024)
    cls_label = torch.rand(2,16)
    print("===> testing model ...")
    model = Model21()
    out = model(data,None, cls_label)
    print(out.shape)

    model = model21H()
    out = model(data, None, cls_label)
    print(out.shape)


