import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class Frequency_based_Token_Selection(nn.Module):
    def __init__(self, keep,stride=16):
        super().__init__()
        self.DWT = DWTForward(J=4, wave='haar', mode='zero').cuda()
        self.IDWT = DWTInverse(wave='haar', mode='zero').cuda()
        self.keep = keep
        self.window_size = stride
        self.stride = stride

    def mask(self, Inverse,window_size=16):
        batch_size, height, width = Inverse.size(0), Inverse.size(-2), Inverse.size(-1)
        Inverse = torch.mean(Inverse, dim=1)
        # create a tensor to store the count of non-zero elements
        count_tensor = torch.zeros((batch_size, height //self.stride, width // self.stride),
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

    def forward(self, x, y, img_path, pattern='a', mode=None, writer=None, step=None):
        Ylx, Yhx = self.DWT(x)  #yhx是列表
        Yly, Yhy = self.DWT(y)
        # You can try to insert the self.show here to reproduce the Fig.4
        low = (Ylx + Yly) / 2
        high = []
        for i in range(len(Yhx)):
            high.append((Yhx[i] + Yhy[i]) / 2)

        Inverse = self.IDWT((low, high))
        selected_tokens_mask = self.mask(Inverse=Inverse, window_size=self.window_size)

        return selected_tokens_mask,Inverse
        
        
    
