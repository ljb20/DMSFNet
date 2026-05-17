import torch
from torch import nn
import torch.nn.functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    # dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx - 2 * torch.matmul(emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


class DCL(nn.Module):
    def __init__(self, num_pos=4, feat_norm='no'):
        super(DCL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self,inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        N = inputs.size(0)
        id_num = N // 2 // self.num_pos

        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
        is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]
        # print(is_neg_c2i.shape)

        centers = []
        for i in range(id_num):
            centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
        centers = torch.stack(centers)

        dist_mat = pdist_torch(centers, inputs)  #  c-i
        # print(dist_mat.shape)
        an = dist_mat * is_neg_c2i
        an = an[an > 1e-6].view(id_num, -1)

        d_neg = torch.mean(an, dim=1, keepdim=True)
        mask_an = (an - d_neg).expand(id_num, N - 2 * self.num_pos).lt(0)  # mask
        an = an * mask_an

        list_an = []
        for i in range (id_num):
            list_an.append(torch.mean(an[i][an[i]>1e-6]))
        an_mean = sum(list_an) / len(list_an)

        ap = dist_mat * ~is_neg_c2i
        ap_mean = torch.mean(ap[ap>1e-6])

        loss = ap_mean / an_mean

        return loss
class DCL2(nn.Module):
    def __init__(self, num_pos=4, feat_norm='no'):
        super(DCL2, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        N = inputs.size(0) // 4
        label1, label2 = targets.chunk(2, 0)

        # 计算样本对之间的距离矩阵
        dist_mat = pdist_torch(inputs, inputs)
        dist_mat_r2g = dist_mat[0:N, 3 * N:4 * N]
        dist_mat_i2g = dist_mat[N:2 * N, 2 * N:3 * N]

        # 样本权重 P_K，假设每个样本权重相等
        P_K = 1 / N

        # 正样本对距离 d_pos
        is_pos = ~label1.expand(N, N).ne(label2.expand(N, N).t())  # 正样本对
        ap_r2g = dist_mat_r2g * is_pos
        ap_i2g = dist_mat_i2g * is_pos
        d_pos = 0.5 * P_K * (torch.sum(ap_r2g[ap_r2g > 1e-6]) + torch.sum(ap_i2g[ap_i2g > 1e-6]))

        # 负样本对距离 d_neg
        is_neg = label1.expand(N, N).ne(label2.expand(N, N).t())  # 负样本对
        an_r2g = dist_mat_r2g * is_neg
        an_i2g = dist_mat_i2g * is_neg

        # 计算阈值 tau
        tau_r2g = torch.mean(an_r2g[an_r2g > 1e-6])
        tau_i2g = torch.mean(an_i2g[an_i2g > 1e-6])

        # 根据 tau 过滤负样本对
        an_r2g_filtered = an_r2g[an_r2g < tau_r2g]
        an_i2g_filtered = an_i2g[an_i2g < tau_i2g]

        d_neg = 0.5 * P_K * (torch.sum(an_r2g_filtered) + torch.sum(an_i2g_filtered))

        # 最终损失
        loss = d_pos / (d_neg + 1e-6)  # 防止分母为 0

        return loss
#  


import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [B, m, d]
      y: pytorch Variable, with shape [B, n, d]
    Returns:
      dist: pytorch Variable, with shape [B, m, n]
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    x = torch.nn.functional.normalize(x, dim=2, p=2)
    y = torch.nn.functional.normalize(y, dim=2, p=2)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(B, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(B, n, m).transpose(-2, -1)
    dist = xx + yy
    dist = dist - 2 * (x @ y.transpose(-2, -1))
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # return 1. / dist
    return dist
    # return -torch.log(dist)


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [B, m, d]
      y: pytorch Variable, with shape [B, n, d]
    Returns:
      dist: pytorch Variable, with shape [B, m, n]
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    x_norm = torch.pow(x, 2).sum(2, keepdim=True).sqrt().expand(B, m, n)
    y_norm = torch.pow(y, 2).sum(2, keepdim=True).sqrt().expand(B, n, m).transpose(-2, -1)
    xy_intersection = x @ y.transpose(-2, -1)
    dist = xy_intersection/(x_norm * y_norm)
    return torch.abs(dist)

class Dissimilar(object):
    def __init__(self, dynamic_balancer=True):
        self.dynamic_balancer = dynamic_balancer
    
    def __call__(self, features):
        B, N, C = features.shape
        dist_mat = cosine_dist(features, features)  # B*N*N
        # dist_mat = euclidean_dist(features, features)
        # 上三角index
        top_triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        _dist = dist_mat[:, top_triu]

        # 1.用softmax替换平均，使得相似度更高的权重更大
        if self.dynamic_balancer:
          weight = F.softmax(_dist, dim=-1)
          dist = torch.mean(torch.sum(weight*_dist, dim=1))
        # 2.直接平均
        else:
          dist = torch.mean(_dist, dim=(0, 1))
        return dist