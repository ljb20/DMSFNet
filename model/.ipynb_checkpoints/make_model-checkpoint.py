from model.vision_transformer import ViT,BlockMask
import torch
import torch.nn as nn
from PIL import Image
from transforms import *
from model.Frequency import Frequency_based_Token_Selection
from model.SFTS import SFTS
from torch.utils.tensorboard import SummaryWriter
from loss.CRM import CRM
import copy
import torch.nn.functional as F
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

class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g =nn.Conv1d(self.high_dim, self.high_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv1d(self.high_dim, self.high_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv1d(self.high_dim, self.high_dim,kernel_size=1, stride=1, padding=0)
            self.W = nn.Linear(self.high_dim, self.high_dim)

    def forward(self, x_l, x_h):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)
        
        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:]).permute(0,2,1)
        W_y = self.W(y)
        z = W_y
        
        return z
class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim)
    def forward(self, x, x0):
        x = x.permute(0,2,1)
        x0 = x0.permute(0,2,1)
        z = self.CNL(x, x0)
        # z = z.permute(0,2,1)
        return z


class build_vision_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer, self).__init__()
        self.in_planes = 768
        self.num_classes = num_classes
        self.base = ViT(img_size=[cfg.H,cfg.W],sie_xishu=3.0,
                        stride_size=cfg.STRIDE_SIZE,num_classes=self.num_classes,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)#加载预训练的ImageNet模型参数，通过调用load_param方法来完成。

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)


    def forward(self, x): 
        if self.training:
            cash_x, attn = self.base(x)
            return cash_x, attn
        else:
            cash_x= self.base(x)
            return cash_x

    def load_param(self, trained_path):#这段代码是用来加载训练好的模型参数的函数。它接收一个已训练模型的路径trained_path，通过torch.load()函数读取该路径下保存的模型参数字典param_dict。
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):#用于加载预训练模型参数进行微调。
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))       
    
class EDITOR(nn.Module):
    def __init__(self, num_classes, cfg):
        super(EDITOR, self).__init__()
        # TWO Modalities share the same backbone
        self.BACKBONE = build_vision_transformer(num_classes, cfg)
        self.num_patches = int(cfg.H // cfg.STRIDE_SIZE[0]) * int(
            cfg.W // cfg.STRIDE_SIZE[1])
        # Ratio means the keep ratio of the patches in each head
        self.ratio = (1 / self.num_patches) * int(cfg.HEAD_KEEP)
        self.SFTS = SFTS(ratio=self.ratio)
        self.l2norm = Normalize()
        self.FREQ_INDEX = Frequency_based_Token_Selection(keep=cfg.FREQUENCY_KEEP,
                                                          stride=cfg.STRIDE_SIZE[0])
        
        self.FUSE_block = BlockMask(num_class=num_classes, dim=self.BACKBONE.in_planes, num_heads=12, mlp_ratio=4.,
                                    qkv_bias=False, momentum=0.8)
        # 用于降低每个模态的特征维度
        self.RGB_REDUCE = nn.Linear(2*self.BACKBONE.in_planes, self.BACKBONE.in_planes)
        self.RGB_REDUCE.apply(weights_init_kaiming)
        self.NIR_REDUCE = nn.Linear(2*self.BACKBONE.in_planes, self.BACKBONE.in_planes)
        self.NIR_REDUCE.apply(weights_init_kaiming)
        # The output learning params of RGB/NIR/TIR cls tokens
        self.BACKBONE_HEAD = nn.Linear(self.BACKBONE.in_planes, num_classes, bias=False)
        self.BACKBONE_HEAD.apply(weights_init_classifier)
        
        self.BACKBONE_BN = nn.BatchNorm1d(self.BACKBONE.in_planes)
        self.BACKBONE_BN.bias.requires_grad_(False)
        self.AL = cfg.AL
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
    
    def forward(self, RGB ,  RGB1 ,NIR , NIR1,  label=None, img_path=None, mode=1,
                writer=None, epoch=1):
        # writer = SummaryWriter(log_dir='/home/ljb/code/EOT/DMSRNet/img_path')
        if self.training:
            if cfg.DATASET == 'sy':    #只有多光谱
                RGB_feat= self.BACKBONE(RGB) #[B,N,C]
                NIR_feat = self.BACKBONE(NIR)
                RGB_feat1 = self.BACKBONE(RGB1) #[B,N,C]
                NIR_feat1 = self.BACKBONE(NIR1)  
                    #原来的三元组损失
                RGB_cls4tri = RGB_feat[:, 0, :]
                RGB_cls4tri1 = RGB_feat1[:, 0, :]
            
                NIR_cls4tri = NIR_feat[:, 0 , :]
                NIR_cls4tri1 = NIR_feat1[:, 0 , :]

                ori = torch.cat([RGB_cls4tri, NIR_cls4tri], dim=0) #[64,768]
                ori1 = torch.cat([RGB_cls4tri1, NIR_cls4tri1], dim=0)

                feats = self.BACKBONE_BN(ori)
                cls_scores = self.BACKBONE_HEAD(feats)

                feats1 = self.BACKBONE_BN(ori1)
                cls_scores1 = self.BACKBONE_HEAD(feats1)
                    
                if self.AL:
                    ori_tri = ori
                    ori_score = cls_scores
                    ori_tri1 = ori1
                    ori_score1 = cls_scores1
                return  ori_score, ori_tri,ori_score1,ori_tri1
            
            else:
                label1,label2=label.chunk(2,0)
                mask_x_fre,x_h = self.FREQ_INDEX(x=RGB, y=RGB1,  img_path=img_path, mode=mode, writer=writer,
                                        step=epoch)
                mask_y_fre,y_h = self.FREQ_INDEX(x=NIR, y=NIR1,  img_path=img_path, mode=mode, writer=writer,
                                        step=epoch)
                RGB_feat, RGB_attn = self.BACKBONE(RGB) #[B,N,C]
                NIR_feat, NIR_attn = self.BACKBONE(NIR)
                RGB_feat1, RGB_attn1 = self.BACKBONE(RGB1) #[B,N,C]
                NIR_feat1, NIR_attn1 = self.BACKBONE(NIR1)
                x_feat,_ = self.BACKBONE((x_h))
                y_feat,_ = self.BACKBONE((y_h))
                
                x_cls = x_feat[:,0,:]
                y_cls = y_feat[:,0,:]
                
                RGB_cls4tri = RGB_feat[:, 0, :] 
                RGB_cls4tri1 = RGB_feat1[:, 0, :] 

                NIR_cls4tri = NIR_feat[:, 0 , :] 
                NIR_cls4tri1 = NIR_feat1[:, 0 , :] 

                ori_tri = torch.cat([RGB_cls4tri, NIR_cls4tri], dim=0) #[64,768]
                ori_tri1 = torch.cat([RGB_cls4tri1, NIR_cls4tri1], dim=0)
                highff = torch.cat([x_cls, y_cls], dim=0)
                
                featff = self.BACKBONE_BN(highff)
                high_score = self.BACKBONE_HEAD(featff)

                feats = self.BACKBONE_BN(ori_tri)
                ori_score = self.BACKBONE_HEAD(feats)

                feats1 = self.BACKBONE_BN(ori_tri1)
                ori_score1 = self.BACKBONE_HEAD(feats1)
                
                
                RGB_feat_s, RGB_feat_s1, mask_x, loss_RGB_bcc = self.SFTS(RGB_feat=RGB_feat,
                                                                            RGB_attn=RGB_attn,
                                                                            NIR_feat=RGB_feat1,
                                                                            NIR_attn=RGB_attn1,
                                                                          
                                                                            img_path=img_path,
                                                                            epoch=epoch, writer=writer,
                                                                            mask_fre=mask_x_fre)

                NIR_feat_s, NIR_feat_s1, mask_y, loss_NIR_bcc = self.SFTS(RGB_feat=NIR_feat,
                                                                            RGB_attn=NIR_attn,
                                                                            NIR_feat=NIR_feat1,
                                                                            NIR_attn=NIR_attn1,
                                                                          
                                                                            img_path=img_path,
                                                                            epoch=epoch, writer=writer,
                                                                            mask_fre=mask_y_fre)
                RGB_feat_s = RGB_feat +RGB_feat_s 
                RGB_feat_s1 = RGB_feat1 +RGB_feat_s1
                NIR_feat_s = NIR_feat +NIR_feat_s
                NIR_feat_s1 = NIR_feat1 +NIR_feat_s1

                feat_s_RGB,loss_ocfr_rgb = self.FUSE_block(RGB_feat_s, RGB_feat_s1, mask=mask_x, label=label1,
                                                    epoch=epoch)
                
                feat_s_NIR,loss_ocfr_nir = self.FUSE_block(NIR_feat_s, NIR_feat_s1, mask=mask_y, label=label2,
                                                    epoch=epoch)
                
                RGB_feat_s = RGB_feat_s + feat_s_RGB[:, :RGB_feat_s.shape[1]]
                RGB_feat_s1 = RGB_feat_s1 + feat_s_RGB[:, RGB_feat_s.shape[1]:]

                NIR_feat_s =NIR_feat_s + feat_s_NIR[:, :NIR_feat_s.shape[1]]
                NIR_feat_s1 =NIR_feat_s1 + feat_s_NIR[:, NIR_feat_s.shape[1]:]
                
                RGB_cls = RGB_feat_s[:,  0, :]
                RGB_cls1 = RGB_feat_s1[:,  0, :]

                NIR_cls = NIR_feat_s[:,  0, :]
                NIR_cls1 = NIR_feat_s1[:,  0, :]
            
                RGB_patch = RGB_feat_s[:,  1:, :]
                RGB_patch1 = RGB_feat_s1[:,  1:, :]

                NIR_patch = NIR_feat_s[:,  1:, :]
                NIR_patch1 = NIR_feat_s1[:,  1:, :]

                def process_patches(patch):
                    row_sum = torch.sum(patch, dim=2)
                    num = (row_sum != 0).sum(dim=1).unsqueeze(-1)
                    patch = torch.sum(patch, dim=1) / num  # [B,C]
                    return patch
                
                RGB_patch = process_patches(RGB_patch)
                RGB_patch1= process_patches(RGB_patch1)
                NIR_patch = process_patches(NIR_patch)
                NIR_patch1= process_patches(NIR_patch1)

                rgb = ( RGB_cls + self.RGB_REDUCE(torch.cat([RGB_cls, RGB_patch], dim=-1)))
                rgb1 = ( RGB_cls1 + self.RGB_REDUCE(torch.cat([RGB_cls1, RGB_patch1], dim=-1)))
                nir = ( NIR_cls + self.NIR_REDUCE(torch.cat([NIR_cls, NIR_patch], dim=-1)))
                nir1 = ( NIR_cls1 + self.NIR_REDUCE(torch.cat([NIR_cls1, NIR_patch1], dim=-1)) )

                cls4t=torch.cat([rgb,nir],dim=0)
                cls4t1=torch.cat([rgb1,nir1],dim=0)
                
                cls_bn = self.BACKBONE_BN(cls4t)
                score=self.BACKBONE_HEAD(cls_bn)

                cls_bn1 = self.BACKBONE_BN(cls4t1)
                score1=self.BACKBONE_HEAD(cls_bn1)
              
                loss_bcc=loss_NIR_bcc+loss_RGB_bcc+loss_ocfr_nir+loss_ocfr_rgb

                if self.AL:
                    return score, cls4t,score1,cls4t1, ori_score, ori_tri,ori_score1,ori_tri1, loss_bcc,high_score,highff
        else:             
            RGB_feat = self.BACKBONE(RGB) 
            #原来的三元组损失
            RGB_cls4tri = RGB_feat[:, 0, :]
            feats = self.BACKBONE_BN(RGB_cls4tri)
            feats = self.l2norm(feats)
            return feats
                
       
            
