import torch
import torch.nn as nn


def index_points(points, idx):
    """
    Index points while keep the data dim as [B,N',C] type

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, M, S]
    Return:
        new_points:, indexed points data, [B, S, C] or [B, M, S, C]
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


def compute_dist_square(src, tar):
    """

    :param src: source point [B, N, C]
    :param tar: target point [B, S, C]
    :return: dis pair [B, N, S]

    dist = (xn - xm)^T*(xn - xm) + (yn - ym)^T*(yn - ym) + (zn - zm)^T*(zn - zm)
         = xn^T*xn+...+2*xn^T*xm+...
    """
    B, N, _ = src.shape
    _, S, _ = tar.shape
    dist = -2 * torch.matmul(src, tar.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(tar ** 2, -1).view(B, 1, S)
    return dist


def farthest_point_sample(point_xyz, n_point):
    """

    :param point_xyz: points xyz coord
    :param n_point: sample num
    :return:
    """

    device = point_xyz.device
    B, N, C = point_xyz.shape

    sample_points_index = torch.zeros([B, n_point], dtype=torch.long).to(device)
    distance = torch.ones([B, N], dtype=torch.long).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for ii in range(n_point):
        sample_points_index[:, ii] = farthest
        sample_point = point_xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((point_xyz - sample_point) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return sample_points_index


def raidus_nn_sample(point_xyz, query_xyz, radius, k):
    """

    :param point_xyz:  all point [B, N, 3]
    :param query_xyz:  query point [B, M, 3]
    :param radius: search radius
    :param k: radius nn points num limit K
    :return: nn idx [B, M, K]
    """
    device = point_xyz.device
    B, N, C = point_xyz.shape
    _, M, _ = query_xyz.shape
    # dist [B, M, N]
    dist = compute_dist_square(point_xyz, query_xyz)
    dist = dist.permute(0, 2, 1)
    # nn_idx [B, M, N]
    nn_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, M, 1])
    nn_idx[dist > radius ** 2] = N

    # sort, make label N lie on the end of each line，leave the first k elements
    nn_idx = torch.sort(nn_idx, dim=-1)[0][:, :, :k]

    # replace non radius nn points use nearest radius nn point
    nn_nearest = nn_idx[:, :, 0].view(B, M, 1).repeat([1, 1, k])
    mask = nn_idx == N
    nn_idx[mask] = nn_nearest[mask]

    return nn_idx


def sample_and_group(points_xyz, points_feature, n_point, radius, nn_k, returnfps=False):
    """


    :param points_xyz: input point xyz [B, N, 3]
    :param points_feature: input point point-wise feature [B, N, D]
    :param n_point: sample points num
    :param radius: nn radius
    :param nn_k: nn k
    :param returnfps: if return  fps_idx
    :return:
        sample_points:xyz [B, n_point, nn_k, 3]
        group_points:norm_xyz+feature [B, npoint, nn_k, 3+D]
    """
    # get shape
    B, N, C = points_xyz.shape
    M = n_point
    # fps sampling
    fps_points_idx = farthest_point_sample(points_xyz, n_point)
    torch.cuda.empty_cache()
    # index points
    sample_points = index_points(points_xyz, fps_points_idx)
    torch.cuda.empty_cache()
    # grouping
    nn_idx = raidus_nn_sample(points_xyz, sample_points, radius, nn_k)
    torch.cuda.empty_cache()
    # index points
    group_points_xyz = index_points(points_xyz, nn_idx)  # [B, n_point, nn_k, C]
    torch.cuda.empty_cache()
    # group normalization
    group_points_xyz_norm = group_points_xyz - sample_points.view(B, M, 1, C)

    # concatenate feature
    if points_feature is not None:
        group_points_featrue = index_points(points_feature, nn_idx)
        group_points = torch.cat([group_points_xyz_norm, group_points_featrue], dim=-1)
    else:
        group_points = group_points_xyz_norm
    if returnfps:
        return sample_points, group_points, group_points_xyz, fps_points_idx
    else:
        return sample_points, group_points


def sample_and_group_all(points_xyz, points_feature):
    """
    Equivalent to sample_and_group with input parameter n_point = 1 ,radius = inf, nn_k = N

    Input:
        points_xyz: input points position data, [B, N, 3]
        points_feature: input points data, [B, N, D]
    Return:
        sample_points: sampled points position data, [B, 1, 3]
        group_points: sampled points data, [B, 1, N, 3+D]
    """

    device = points_xyz.device
    B, N, C = points_xyz.shape

    # sample point is [0, 0, 0]
    sample_points = torch.zeros(B, 1, C).to(device)

    # grouping all points
    group_points_xyz = points_xyz.view(B, 1, N, C)
    if points_feature is not None:
        group_points = torch.cat([group_points_xyz, points_feature.view(B, 1, N, -1)], dim=-1)
    else:
        group_points = group_points_xyz
    return sample_points, group_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_points, radius, nn_k, in_channel, mlp, group_all=False):
        """

        :param n_points: sample points num
        :param radius: nn radius
        :param nn_k: nn num
        :param in_channel: input channel
        :param mlp: pointnet mlp
        :param group_all: if group all point
        """
        super(PointNetSetAbstraction, self).__init__()
        self.n_point = n_points
        self.radius = radius
        self.nn_k = nn_k
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relus = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.bns.append(nn.BatchNorm2d(out_channel))
            self.relus.append(nn.ReLU())
            last_channel = out_channel

    def forward(self, points_xyz, points_features):
        """

        :param point_xyz: [B, C, N]
        :param point_features: [B, D, N]
        :return:
            out_xyz: [B, C, M]
            out_features: [B, D', M]
        """

        points_xyz = points_xyz.permute(0, 2, 1)
        if points_features is not None:
            points_features = points_features.permute(0, 2, 1)

        if not self.group_all:
            out_xyz, group_points = sample_and_group(points_xyz, points_features, self.n_point, self.radius, self.nn_k)
        else:
            out_xyz, group_points = sample_and_group_all(points_xyz, points_features)

        group_points = group_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        out_xyz = out_xyz.permute(0, 2, 1)
        x = group_points
        for mpl_conv, bn, relu in zip(self.mlp_convs, self.bns, self.relus):
            x = mpl_conv(x)
            x = bn(x)
            x = relu(x)

        x = torch.max(x, dim=2)[0]
        return out_xyz, x


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, n_points, radius_list, nn_k_list, in_channel, mlp_list):
        """

        :param n_points: sample num
        :param radius_list: radius list [r1,r2,...]
        :param nn_k_list: nn_k list [n1,n2,...]
        :param in_channel: input channel
        :param mlp_list: pointnet mlp for each scale
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.n_point = n_points
        self.radius_list = radius_list
        self.nn_k_list = nn_k_list
        self.mlp_convs_blocks = nn.ModuleList()
        self.bns_blocks = nn.ModuleList()
        self.relus_blocks = nn.ModuleList()
        for mlp in mlp_list:
            last_channel = in_channel + 3
            mlp_convs = nn.ModuleList()
            bns = nn.ModuleList()
            relus = nn.ModuleList()
            for out_channel in mlp:
                mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                relus.append(nn.ReLU())
                last_channel = out_channel
            self.mlp_convs_blocks.append(mlp_convs)
            self.bns_blocks.append(bns)
            self.relus_blocks.append(relus)

    def forward(self, points_xyz, points_features):
        """

        :param point_xyz: [B, C, N]
        :param point_features: [B, D, N]
        :return:
            out_xyz: [B, C, M]
            out_features: [B, D', M]
        """

        points_xyz = points_xyz.permute(0, 2, 1)
        if points_features is not None:
            points_features = points_features.permute(0, 2, 1)
        B, N, C = points_xyz.shape
        S = self.n_point
        out_xyz = index_points(points_xyz, farthest_point_sample(points_xyz, self.n_point))
        x_list = []
        for radius, nn_k, mlp_convs, bns, relus in zip(self.radius_list, self.nn_k_list, self.mlp_convs_blocks,
                                                       self.bns_blocks, self.relus_blocks):
            nn_idx = raidus_nn_sample(points_xyz, out_xyz, radius, nn_k)

            # index points
            group_points_xyz = index_points(points_xyz, nn_idx)  # [B, n_point, nn_k, C]

            # group normalization
            group_points_xyz_norm = group_points_xyz - out_xyz.view(B, S, 1, C)

            # concatenate feature
            if points_features is not None:
                group_points_featrue = index_points(points_features, nn_idx)
                group_points = torch.cat([group_points_xyz_norm, group_points_featrue], dim=-1)
            else:
                group_points = group_points_xyz_norm

            group_points = group_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
            x = group_points
            for mpl_conv, bn, relu in zip(mlp_convs, bns, relus):
                x = mpl_conv(x)
                x = bn(x)
                x = relu(x)

            x = torch.max(x, dim=2)[0]
            x_list.append(x)
        out_xyz = out_xyz.permute(0, 2, 1)
        x = torch.cat(x_list, dim=1)
        return out_xyz, x
