import torch
import torch.nn as nn
import pywt
import pywt.data
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
from config.config import cfg
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print(x_HH[:, 0, :, :])
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width])

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt_init(x)


# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W] could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y


# Channel Attention Layer
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Half Wavelet Attention Block (HWAB)
class HWAB(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size=3, reduction=4, bias=False, act=nn.PReLU()):
        super(HWAB, self).__init__()
        self.dwt = DWT().cuda()
        self.iwt = IWT().cuda()
        modules_body = \
            [
                conv(n_feat * 4, n_feat, kernel_size, bias=bias),
                act,
                conv(n_feat, n_feat * 4, kernel_size, bias=bias)
            ]
        self.body = nn.Sequential(*modules_body)

        self.WSA = SALayer()
        self.WCA = CALayer(n_feat * 4, reduction, bias=bias)
        self.conv1x1 = nn.Conv2d(n_feat * 8, n_feat * 4, kernel_size=1, bias=bias)
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=False)

    def forward(self, x):
        x_dwt = self.dwt(x)  # ([16, 12, 128, 64])
        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x_dwt
        wavelet_path = self.iwt(res)  # ([16, 3, 256, 128])
        wavelet_path = wavelet_path.to('cuda:0')
        out = wavelet_path
        return out


class Frequency_based_Token_Selection(nn.Module):
    def __init__(self, keep, stride=16):
        super().__init__()
        self.DWT = DWTForward(J=4, wave='haar', mode='zero').cuda()
        self.IDWT = DWTInverse(wave='haar', mode='zero').cuda()
        self.keep = keep
        self.window_size = stride
        self.stride = stride
        if cfg.DATASET == 'sysu':
            self.block = HWAB(n_feat=3, o_feat=3).cuda()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=False)
    def mask(self, Inverse, window_size=16):
        batch_size, height, width = Inverse.size(0), Inverse.size(-2), Inverse.size(-1)
        Inverse = torch.mean(Inverse, dim=1)
        # create a tensor to store the count of non-zero elements
        count_tensor = torch.zeros((batch_size, height // self.stride, width // self.stride),
                                   dtype=torch.int).cuda()
        # For each image in the batch
        for batch_idx in range(batch_size):
            image = Inverse[batch_idx]  # 获取当前图像
            # With a sliding window, unfold the image into a tensor
            unfolded = F.unfold(image.unsqueeze(0).unsqueeze(0), window_size, stride=self.stride)
            # Turns elements greater than 0 into binary, then sums to count the number of elements greater than 0
            count = unfolded.gt(0).sum(1)
            count = count.view(height // self.stride, width // self.stride)
            count_tensor[batch_idx] = count

            # Get the index of the maximum value of each image
        _, topk_indices = torch.topk(count_tensor.flatten(1), int(self.keep), dim=1)
        topk_indices = torch.sort(topk_indices, dim=1).values
        selected_tokens_mask = torch.zeros((batch_size, (height // self.stride) * (width // self.stride)),
                                           dtype=torch.bool).cuda()
        selected_tokens_mask.scatter_(1, topk_indices, 1)
        return selected_tokens_mask

    def forward(self, x, y, img_path, pattern='a', mode=None, writer=None, step=None, z=None):
        Ylx, Yhx = self.DWT(x)  # yhx是列表
        Yly, Yhy = self.DWT(y)
        # You can try to insert the self.show here to reproduce the Fig.4
        low = (Ylx + Yly) / 2
        high = []
        x = self.IDWT((Ylx, Yhx))
        y = self.IDWT((Yly, Yhy))
        for i in range(len(Yhx)):
            high.append((Yhx[i] + Yhy[i]) / 2)
        if cfg.DATASET == 'sysu':
            out_x = self.conv1(x)
            out_y = self.conv1(y)
            Inverse = self.IDWT((low, high)) + out_x + out_y
        else:
            Inverse = self.IDWT((low, high))

        selected_tokens_mask = self.mask(Inverse=Inverse , window_size=self.window_size)
       
        return selected_tokens_mask, Inverse