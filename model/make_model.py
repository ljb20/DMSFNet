from model.vision_transformer import ViT, BlockMask
import torch
import torch.nn as nn
from PIL import Image
from transforms import *
from model.Frequency import Frequency_based_Token_Selection
from model.SFTS import SFTS
from torch.utils.tensorboard import SummaryWriter
import copy
import torch.nn.functional as F
from torch.nn import init
from loss.Triplet import QuarCenterTripletLoss
from model.OCFR import OCFR
from NFCC import NFC
from loss.DCL import Dissimilar


# L2 norm
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_vision_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer, self).__init__()
        self.in_planes = 768
        self.num_classes = num_classes

        self.base = ViT(img_size=[cfg.H, cfg.W], sie_xishu=3.0,
                        stride_size=cfg.STRIDE_SIZE, num_classes=self.num_classes,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE,
                        cfg=cfg)

        self.base.load_param(cfg.PRETRAIN_PATH)  # 加载预训练的ImageNet模型参数，通过调用load_param方法来完成。

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, x):
        cash_x, attn = self.base(x)
        return cash_x, attn

    def load_param(self,
                   trained_path):  # 这段代码是用来加载训练好的模型参数的函数。它接收一个已训练模型的路径trained_path，通过torch.load()函数读取该路径下保存的模型参数字典param_dict。
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):  # 用于加载预训练模型参数进行微调。
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class EDITOR(nn.Module):
    def __init__(self, num_classes, cfg):
        super(EDITOR, self).__init__()

        self.BACKBONE = build_vision_transformer(num_classes, cfg)
        self.num_patches = int(cfg.H // cfg.STRIDE_SIZE[0]) * int(
            cfg.W // cfg.STRIDE_SIZE[1])
        # Ratio means the keep ratio of the patches in each head
        self.ratio = (1 / self.num_patches) * int(cfg.HEAD_KEEP)
        self.SFTS = SFTS(ratio=self.ratio)
        self.l2norm = Normalize(2)
        self.FREQ_INDEX = Frequency_based_Token_Selection(keep=cfg.FREQUENCY_KEEP,
                                                          stride=cfg.STRIDE_SIZE[0])

        self.FUSE_block = BlockMask(num_class=num_classes, dim=self.BACKBONE.in_planes, num_heads=12, mlp_ratio=4.,
                                    qkv_bias=False, momentum=0.8)

        # 用于降低每个模态的特征维度
        self.RGB_REDUCE = nn.Linear(2 * self.BACKBONE.in_planes, self.BACKBONE.in_planes)
        self.RGB_REDUCE.apply(weights_init_kaiming)
        self.NIR_REDUCE = nn.Linear(2 * self.BACKBONE.in_planes, self.BACKBONE.in_planes)
        self.NIR_REDUCE.apply(weights_init_kaiming)
        self.BACKBONE_HEAD = nn.Linear(self.BACKBONE.in_planes, num_classes, bias=False)
        self.BACKBONE_HEAD.apply(weights_init_classifier)

        self.BACKBONE_BN = nn.BatchNorm1d(self.BACKBONE.in_planes)
        self.BACKBONE_BN.bias.requires_grad_(False)

        self.AL = cfg.AL
        self.cri_tri = QuarCenterTripletLoss(k_size=4, margin=0.1)
        self.memory_cls = OCFR(dim=768, num_class=num_classes, momentum=0.8)
        self.dissimilar = Dissimilar(dynamic_balancer=True)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, RGB, RGB1, NIR, NIR1, label=None, img_path=None, mode=1,
                writer=None, epoch=1):

        if self.training:
            if cfg.guangpu == 1:
                RGB_feat, RGB_attn = self.BACKBONE(RGB)  # [B,N,C]
                NIR_feat, NIR_attn = self.BACKBONE(NIR)

                RGB_feat1, RGB_attn1 = self.BACKBONE(RGB1)  # [B,N,C]
                NIR_feat1, NIR_attn1 = self.BACKBONE(NIR1)

                RGB_cls4tri = RGB_feat[:, 0, :]
                NIR_cls4tri = NIR_feat[:, 0, :]

                RGB_cls4tri1 = RGB_feat1[:, 0, :]
                NIR_cls4tri1 = NIR_feat1[:, 0, :]

                ori_tri = torch.cat([RGB_cls4tri, NIR_cls4tri], dim=0)  # [64,768]
                ori_tri1 = torch.cat([RGB_cls4tri1, NIR_cls4tri1], dim=0)

                ori_score = self.BACKBONE_HEAD(self.BACKBONE_BN(ori_tri))
                ori_score1 = self.BACKBONE_HEAD(self.BACKBONE_BN(ori_tri1))
                return ori_score, ori_tri, ori_score1, ori_tri1

            else:

                label1, label2 = label.chunk(2, 0)
                mask_x_fre, x_h = self.FREQ_INDEX(x=RGB, y=RGB1, z=None, img_path=img_path, mode=mode, writer=writer,
                                                  step=epoch)
                mask_y_fre, y_h = self.FREQ_INDEX(x=NIR, y=NIR1, z=None, img_path=img_path, mode=mode, writer=writer,
                                                  step=epoch)  # 16,162

                RGB_feat, RGB_attn = self.BACKBONE(RGB)  # [B,N,C]
                NIR_feat, NIR_attn = self.BACKBONE(NIR)

                RGB_feat1, RGB_attn1 = self.BACKBONE(RGB1)  # [B,N,C]
                NIR_feat1, NIR_attn1 = self.BACKBONE(NIR1)

                RGB_cls4tri = RGB_feat[:, 0, :]
                NIR_cls4tri = NIR_feat[:, 0, :]

                RGB_cls4tri1 = RGB_feat1[:, 0, :]
                NIR_cls4tri1 = NIR_feat1[:, 0, :]

                ori_tri = torch.cat([RGB_cls4tri, NIR_cls4tri], dim=0)  # [64,768]
                ori_tri1 = torch.cat([RGB_cls4tri1, NIR_cls4tri1], dim=0)

                ori_score = self.BACKBONE_HEAD(self.BACKBONE_BN(ori_tri))
                ori_score1 = self.BACKBONE_HEAD(self.BACKBONE_BN(ori_tri1))

                RGB_feat_s, RGB_feat_s1, mask_x, loss_RGB_bcc = self.SFTS(RGB_feat=RGB_feat,
                                                                          RGB_attn=RGB_attn,
                                                                          NIR_feat=RGB_feat1,
                                                                          NIR_attn=RGB_attn1,
                                                                          img_path=img_path,
                                                                          label=label1,
                                                                          epoch=epoch, writer=writer,
                                                                          mask_fre=mask_x_fre)

                NIR_feat_s, NIR_feat_s1, mask_y, loss_NIR_bcc = self.SFTS(RGB_feat=NIR_feat,
                                                                          RGB_attn=NIR_attn,
                                                                          NIR_feat=NIR_feat1,
                                                                          NIR_attn=NIR_attn1,
                                                                          img_path=img_path,
                                                                          label=label1,
                                                                          epoch=epoch, writer=writer,
                                                                          mask_fre=mask_y_fre)

                RGB_feat_s = RGB_feat + RGB_feat_s
                RGB_feat_s1 = RGB_feat1 + RGB_feat_s1
                NIR_feat_s = NIR_feat + NIR_feat_s
                NIR_feat_s1 = NIR_feat1 + NIR_feat_s1

                RGB_cls = RGB_feat_s[:, 0, :]
                RGB_cls1 = RGB_feat_s1[:, 0, :]

                NIR_cls = NIR_feat_s[:, 0, :]
                NIR_cls1 = NIR_feat_s1[:, 0, :]

                RGB_patch = RGB_feat_s[:, 1:, :]
                RGB_patch1 = RGB_feat_s1[:, 1:, :]

                NIR_patch = NIR_feat_s[:, 1:, :]
                NIR_patch1 = NIR_feat_s1[:, 1:, :]

                loss_ocfr_rgb = self.memory_cls(RGB_cls, RGB_cls1, label_=label1, epoch=epoch)
                loss_ocfr_nir = self.memory_cls(NIR_cls, NIR_cls1, label_=label2, epoch=epoch)
                loss_tri2 = self.cri_tri(torch.cat([RGB_cls, RGB_cls1, NIR_cls, NIR_cls1]),
                                         torch.cat([label1, label1, label1, label1]))[0]

                def process_patches(patch):
                    row_sum = torch.sum(patch, dim=2)
                    num = (row_sum != 0).sum(dim=1).unsqueeze(-1)
                    patch = torch.sum(patch, dim=1) / num  # [B,C]
                    return patch

                RGB_patch = process_patches(RGB_patch)
                RGB_patch1 = process_patches(RGB_patch1)
                NIR_patch = process_patches(NIR_patch)
                NIR_patch1 = process_patches(NIR_patch1)

                rgb = RGB_cls + self.RGB_REDUCE(torch.cat([RGB_cls, RGB_patch], dim=-1))
                rgb1 = RGB_cls1 + self.RGB_REDUCE(torch.cat([RGB_cls1, RGB_patch1], dim=-1))
                nir = NIR_cls + self.NIR_REDUCE(torch.cat([NIR_cls, NIR_patch], dim=-1))
                nir1 = NIR_cls1 + self.NIR_REDUCE(torch.cat([NIR_cls1, NIR_patch1], dim=-1))

                cls4t = torch.cat([rgb, nir], dim=0)
                cls4t1 = torch.cat([rgb1, nir1], dim=0)

                score = self.BACKBONE_HEAD(self.BACKBONE_BN(cls4t))
                score1 = self.BACKBONE_HEAD(self.BACKBONE_BN(cls4t1))

                loss_bcc = loss_NIR_bcc + loss_RGB_bcc + loss_ocfr_nir + loss_ocfr_rgb

                if self.AL:
                    return score, cls4t, score1, cls4t1, ori_score, ori_tri, ori_score1, ori_tri1, loss_bcc, loss_tri2


        else:
            RGB_feat, _ = self.BACKBONE(RGB)
            RGB_cls4tri = RGB_feat[:, 0]
            feats = self.BACKBONE_BN(RGB_cls4tri)
            feats = self.l2norm(feats)
            feats = NFC(feats)
            return feats