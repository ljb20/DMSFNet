import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y, eps=1e-12):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # dist.addmm_(x, y.t(), beta=1, alpha=-2) #dist.addmm_(1, -2, x, y.t())
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=eps).sqrt()

    return dist


def hard_example_mining(dist_mat, target):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = target.expand(N, N).eq(target.expand(N, N).t())
    is_neg = target.expand(N, N).ne(target.expand(N, N).t())

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)

    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    return dist_ap, dist_an

def hard_example_mining2(dist_mat, target):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    label1,label2 = target.chunk(2,0)
    # shape [N, N]
    is_pos = label1.expand(N, N).eq(label2.expand(N, N).t())
    is_neg = label1.expand(N, N).ne(label2.expand(N, N).t())

    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)

    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    return dist_ap, dist_an


class TripletLoss(nn.Module):
    def __init__(self, margin, feat_norm='yes'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.feat_norm = feat_norm
        if margin >= 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, global_feat1, global_feat2, target):
        if self.feat_norm == 'yes':
            global_feat1 = F.normalize(global_feat1, p=2, dim=-1)
            global_feat2 = F.normalize(global_feat2, p=2, dim=-1)

        dist_mat = euclidean_dist(global_feat1, global_feat2)
        dist_ap, dist_an = hard_example_mining(dist_mat, target)

        y = dist_an.new().resize_as_(dist_an).fill_(1)  # 创建一个与dist_an相同形状的张量y，并填充为1。
        if self.margin >= 0:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx= dist_mtx - 2 * torch.matmul(emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


class QuarCenterTripletLoss(nn.Module):
    def __init__(self, k_size, margin=0, t=0):
        super(QuarCenterTripletLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs ,targets):
        n = inputs.size(0)

        targetsIND = targets[0:n // 2]
        inputsRGBD = inputs[0:n // 2]
        targetsIND = targets[n // 2:n]
        inputsIRD = inputs[n // 2:n]
        targets_all = targets
        # Come to centers
        # rgb1和ir1---rgb2和ir2
        centers = []
        center_IR = []
        center_RGB = []
        centers_dualIR1 = []
        centers_dualIR2 = []
        centers_dualRGB1 = []
        centers_dualRGB2 = []

        for i in range(n):
            centers.append(inputs[targets_all == targets_all[i]].mean(0))
        centers = torch.stack(centers)
     
        targetsIN = targets[0:n // 4]
        targetsRGB = targets[3 * n // 4:n]
        targets = torch.cat([targetsIN,targetsRGB])

        inputsRGB1 = inputs[0:n // 4]
        inputsRGB2 = inputs[n // 4:n // 2]
        inputsIR1 = inputs[n // 2:3 * n // 4]
        inputsIR2 = inputs[3 * n // 4:n]

        for i in range(n // 4):
            # quadruple
            centers_dualIR1.append(inputsIR1[targetsIN == targetsIN[i]].mean(0))
            centers_dualIR2.append(inputsIR2[targetsIN == targetsIN[i]].mean(0))
            centers_dualRGB1.append(inputsRGB1[targetsIN == targetsIN[i]].mean(0))
            centers_dualRGB2.append(inputsRGB2[targetsIN == targetsIN[i]].mean(0))
        
        for i in range(n // 2):
            center_IR.append(inputsIRD[targetsIND == targetsIND[i]].mean(0))
        for i in range(n // 2):
            center_RGB.append(inputsRGBD[targetsIND == targetsIND[i]].mean(0))
        center_IR = torch.stack(center_IR)
        center_RGB = torch.stack(center_RGB)
        centers_dualIR1 = torch.stack(centers_dualIR1)  # IR1
        centers_dualIR2 = torch.stack(centers_dualIR2)  # IR2   
        centers_dualRGB1 = torch.stack(centers_dualRGB1)  # RGB1
        centers_dualRGB2 = torch.stack(centers_dualRGB2)  # RGB2

       

        def tri_loss(inputsRGBD,center_IR,targets):
            dist = euclidean_dist(inputsRGBD,center_IR)
            # For each anchor, find the hardest positive and negative
            dist_ap, dist_an = hard_example_mining2(dist, targets)
            y = dist_an.new().resize_as_(dist_an).fill_(1)  # 创建一个与dist_an相同形状的张量y，并填充为1。
            if self.margin >= 0:
                loss = self.ranking_loss(dist_an, dist_ap, y)
            else:
                loss = self.ranking_loss(dist_an - dist_ap, y)
            return loss
        def tri_loss2(inputsRGBD,center_IR,targets):
            dist = euclidean_dist(inputsRGBD,center_IR)
          
            # For each anchor, find the hardest positive and negative
            dist_ap, dist_an = hard_example_mining(dist, targets)
            y = dist_an.new().resize_as_(dist_an).fill_(1)  # 创建一个与dist_an相同形状的张量y，并填充为1。
            if self.margin >= 0:
                loss = self.ranking_loss(dist_an, dist_ap, y)
            else:
                loss = self.ranking_loss(dist_an - dist_ap, y)
            return loss
        loss1 = tri_loss(inputsIR1,centers_dualRGB1,targets)+tri_loss(inputsRGB1,centers_dualIR1,targets)\
            +tri_loss(inputsIR2,centers_dualRGB1,targets)+tri_loss(inputsRGB2,centers_dualIR1,targets)
           

        loss2 = tri_loss(inputsIR2,centers_dualRGB1,targets)+tri_loss(inputsRGB2,centers_dualIR1,targets)
            

        loss3 = tri_loss2(inputs,centers,targets_all)       
    
        return loss1,loss2,loss3