from dataloader import SYSUData, RegDBData,TestData, GenIdx, IdentitySampler,LLCMData
from datamanager import process_gallery_sysu, process_query_sysu, process_test_regdb,process_gallery_llcm,process_query_llcm
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
import torch
from torch.cuda import amp
import torch.nn as nn
import os.path as osp
import os
from model.make_model import EDITOR
import time
import optimizer
from scheduler import create_scheduler
from loss.Triplet import TripletLoss,QuarCenterTripletLoss
from loss.MSEL import MSEL
from loss.DCL import DCL
from utils import *
import transforms
import sys
from optimizer import make_optimizer
from config.config import cfg
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
import argparse

parser = argparse.ArgumentParser(description="PMT Training")
parser.add_argument('--config_file', default='config/SYSU.yml',
                    help='path to config file', type=str)
parser.add_argument('--trial', default=1,
                    help='only for RegDB', type=int)
parser.add_argument('--resume', '-r', default='',
                    help='resume from checkpoint', type=str)
parser.add_argument('--model_path', default='save_model/',
                    help='model save path', type=str)
parser.add_argument('--log_path', default='log/', type=str, 
                    help='log save path')
parser.add_argument('--num_workers', default=0,
                    help='number of data loading workers', type=int)
parser.add_argument('--start_test', default=0,
                    help='start to test in training', type=int)
parser.add_argument('--test_batch', default=64,
                    help='batch size for test', type=int)
parser.add_argument('--test_epoch', default=2,
                    help='test model every 2 epochs', type=int)
parser.add_argument('--save_epoch', default=2,
                    help='save model every 2 epochs', type=int)
parser.add_argument('--gpu', default='0',
                    help='gpu device ids for CUDA_VISIBLE_DEVICES', type=str)
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None,nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.config_file != '':
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

set_seed(cfg.SEED)#这是为了保证结果的可重复性，在训练过程中生成的随机数将具有相同的顺序。
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = transforms.test_transforms(
    cfg.H, cfg.W, normalize)
transform_color1 = transforms.train_transforms_color1(
    cfg.H, cfg.W, normalize)
transform_color2 = transforms.train_transforms_color2(
    cfg.H, cfg.W, normalize)
transform_thermal1 = transforms.train_transforms_thermal1(
    cfg.H, cfg.W, normalize)
transform_thermal2 = transforms.train_transforms_thermal2(
    cfg.H, cfg.W, normalize)
train_transforms_gray = transforms.train_transforms_gray(
    cfg.H, cfg.W, normalize)
transform_train = transform_color1, transform_color2, transform_thermal1, transform_thermal2
transform_train1 = transform_color1, train_transforms_gray,transform_thermal1,train_transforms_gray


if cfg.DATASET == 'sysu':
    data_path = cfg.DATA_PATH_SYSU
    log_path = args.log_path + 'sysu_log/A300/'  
    trainset_gray = SYSUData(data_path, transform=transform_train1)
    color_pos_gray, thermal_pos_gray = GenIdx(trainset_gray.train_color_label, trainset_gray.train_thermal_label)
    trainset_rgb = SYSUData(data_path, transform=transform_train)
    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label)
    
elif cfg.DATASET == 'regdb':
    data_path = cfg.DATA_PATH_RegDB
    log_path = args.log_path + 'regdb_log/'
    trainset_gray = RegDBData(data_path, args.trial, transform=transform_train1)
    color_pos_gray, thermal_pos_gray = GenIdx(trainset_gray.train_color_label, trainset_gray.train_thermal_label)
    trainset_rgb = RegDBData(data_path, args.trial,transform=transform_train)
    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label)
    print('Current trial:', args.trial)
    print('Current ff:',cfg.FREQUENCY_KEEP)

elif cfg.DATASET == 'llcm':
    data_path = cfg.DATA_PATH_LLCM
    log_path = args.log_path + 'llcm_log/A300/'
    test_mode = [1,2]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;
    trainset_gray = LLCMData(data_path, args.trial, transform=transform_train1)
    color_pos_gray, thermal_pos_gray = GenIdx(trainset_gray.train_color_label, trainset_gray.train_thermal_label)
    trainset_rgb = LLCMData(data_path, args.trial,transform=transform_train)
    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label)
    print('Current trial:', args.trial)
    print('Current ff:',cfg.FREQUENCY_KEEP)

num_classes = len(np.unique(trainset_rgb.train_color_label))
model = EDITOR(num_classes=num_classes,cfg = cfg)
model.to(device)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
suffix = cfg.DATASET
sys.stdout = Logger(log_path + suffix + '_os.txt')
# load checkpoint
if len(args.resume) > 0:
    model_path = args.model_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        model.load_param(model_path)
        print('==> loaded checkpoint {}'.format(args.resume))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# Loss
criterion_ID = nn.CrossEntropyLoss()
criterion_Tri = TripletLoss(margin=cfg.MARGIN, feat_norm='no')
criterion_DCL = DCL(num_pos=cfg.NUM_POS, feat_norm='no')
cri_tri = QuarCenterTripletLoss(k_size=4, margin=0.1)
criterion_MSEL = MSEL(num_pos=cfg.NUM_POS, feat_norm='no')
optimizer = make_optimizer(cfg, model)
scheduler = create_scheduler(cfg, optimizer)

scaler = amp.GradScaler()

if cfg.DATASET == 'sysu':   # for test
    query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))

    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0, gall_mode='single')
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))

elif cfg.DATASET == 'regdb':
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))

elif cfg.DATASET == 'llcm':
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))

# Test loader
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,drop_last=True)
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,drop_last=True)

loss_meter = AverageMeter()
loss_ce_meter = AverageMeter()
loss_tri_meter = AverageMeter()
acc_rgb_meter = AverageMeter()
acc_ir_meter = AverageMeter()
loss_smsm_meter=AverageMeter()

def train(epoch):
    start_time = time.time()

    loss_meter.reset()
    loss_ce_meter.reset()
    loss_tri_meter.reset()
    acc_rgb_meter.reset()
    acc_ir_meter.reset()
    loss_smsm_meter.reset()
    scheduler.step(epoch)#调用学习率调度器的step方法，用于更新当前epoch的学习率。
    model.train()

    for idx, (input1, input2,input1_1,input2_2, label1, label2) in enumerate(trainloader):

        optimizer.zero_grad()
        input1 = input1.to(device)
        input2 = input2.to(device)
        input1_1 = input1_1.to(device)
        input2_2 = input2_2.to(device)
       
        label1 = label1.to(device)
        label2 = label2.to(device)
        labels = torch.cat((label1,label2),0)
        labs = torch.cat((label1,label2,label1,label2),0)
        with amp.autocast(enabled=True):
            if cfg.guangpu ==1:       #只有多光谱，用来做消融实验的
                ori_score, ori_tri,ori_score1,ori_tri1= model(input1, input2,input1_1,input2_2,label=labels)
                ori_score_1,ori_score_2 = ori_score.chunk(2,0)
                ori_score1_1,ori_score1_2 = ori_score1.chunk(2,0)
                loss_id=(criterion_ID(ori_score_1,label1)+criterion_ID(ori_score_2,label2)\
                        +criterion_ID(ori_score1_1,label1)+criterion_ID(ori_score1_2,label2))/2
                ori_tri_1,ori_tri_2 = ori_tri.chunk(2,0)
                ori_tri1_1,ori_tri1_2 = ori_tri1.chunk(2,0)
               
                if cfg.METHOD == 'PMT':
                    if epoch <= cfg.PL_EPOCH :
                        loss_tri = (criterion_Tri(ori_tri_1, ori_tri_1, label1)+criterion_Tri(ori_tri_2, ori_tri_2, label2)\
                                    +criterion_Tri(ori_tri1_1, ori_tri1_1, label1)+criterion_Tri(ori_tri1_2, ori_tri1_2, label2))/2            
                        
                        loss = loss_id + loss_tri+cfg.cmsm*loss_cmsm

                    else:
                        loss_dcl = (criterion_DCL(ori_tri, labels))
                        loss_msel = (criterion_MSEL(ori_tri, labels))
                        loss_tri = (criterion_Tri(ori_tri, ori_tri, labels)+criterion_Tri(ori_tri1, ori_tri1, labels))/2

                        loss = loss_id + loss_tri +cfg.DCL*loss_dcl+cfg.MSEL*loss_msel

            else :

                score, cls4t,score1,cls4t1, ori_score, ori_tri,ori_score1,ori_tri1,loss_bcc,loss_tri2\
                 = model(input1, input2,input1_1,input2_2,label=labels)
                score_1,score_2=score.chunk(2,0)
                score1_1,score1_2=score1.chunk(2,0)
                ori_score_1,ori_score_2 = ori_score.chunk(2,0)
                ori_score1_1,ori_score1_2 = ori_score1.chunk(2,0)

                loss_id=(criterion_ID(score_1,label1)+criterion_ID(score_2,label2)\
                        +criterion_ID(score1_1,label1)+criterion_ID(score1_2,label2)\
                        +criterion_ID(ori_score_1,label1)+criterion_ID(ori_score_2,label2)\
                        +criterion_ID(ori_score1_1,label1)+criterion_ID(ori_score1_2,label2))/4
         
                cls4t_1,cls4t_2=cls4t.chunk(2,0)
                cls4t1_1,cls4t1_2=cls4t1.chunk(2,0)
                ori_tri_1,ori_tri_2 = ori_tri.chunk(2,0)
                ori_tri1_1,ori_tri1_2 = ori_tri1.chunk(2,0)

                quai_ori = torch.cat([ori_tri_1,ori_tri1_1,ori_tri_2,ori_tri1_2])
                quai_ori1 = torch.cat([cls4t_1,cls4t1_1,cls4t_2,cls4t1_2])
                loss_tri1,_ ,_= cri_tri(quai_ori,labs)
                _, _, loss_tri3= cri_tri(quai_ori1,labs)
                
                if cfg.METHOD == 'PMT':
                    if epoch <= cfg.PL_EPOCH :
                        loss_tri = ( criterion_Tri(cls4t_1, cls4t_1, label1)+criterion_Tri(cls4t_2, cls4t_2, label2)\
                                    +criterion_Tri(cls4t1_1, cls4t1_1, label1)+criterion_Tri(cls4t1_2, cls4t1_2, label2)\
                                    +criterion_Tri(ori_tri_1, ori_tri_1, label1)+criterion_Tri(ori_tri_2, ori_tri_2, label2)\
                                    +criterion_Tri(ori_tri1_1, ori_tri1_1, label1)+criterion_Tri(ori_tri1_2, ori_tri1_2, label2))/4

                        loss_tri= loss_tri +cfg.cmsm*loss_tri1 +cfg.tri2*loss_tri2 
                 
                        loss = loss_id + loss_tri +loss_bcc
                    else:
                        loss_dcl = (criterion_DCL(ori_tri, labels))
                        loss_msel = (criterion_MSEL(ori_tri, labels))
                        loss_tri = (criterion_Tri(cls4t, cls4t, labels)+criterion_Tri(cls4t1, cls4t1, labels)\
                            +criterion_Tri(ori_tri, ori_tri, labels)+criterion_Tri(ori_tri1, ori_tri1, labels))/4

                        loss_tri = loss_tri  +cfg.cmsm*loss_tri1 +cfg.tri2*loss_tri2 + loss_tri3

                        loss = loss_id  +loss_tri+cfg.DCL*loss_dcl+cfg.MSEL*loss_msel  +loss_bcc

          

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_tri_meter.update(loss_tri.item())
        loss_ce_meter.update(loss_id.item())
        loss_meter.update(loss.item())

        torch.cuda.synchronize()

        if (idx + 1) % 32 == 0 :
            print('Epoch[{}] Iteration[{}/{}]'
                  ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                  'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                  'Base Lr: {:.2e} '.format(epoch, (idx+1),
                len(trainloader), loss_meter.avg, loss_tri_meter.avg,
                loss_ce_meter.avg, acc_rgb_meter.avg, acc_ir_meter.avg,
                optimizer.state_dict()['param_groups'][0]['lr']))

    end_time = time.time()
    time_per_batch = end_time - start_time
    print(' Epoch {} done. Time per batch: {:.1f}[min] '.format(epoch, time_per_batch/60))


def test(query_loader, gall_loader, dataset = 'sysu'):
    model.eval()
    nquery = len(query_label)
    ngall = len(gall_label)
    print('Testing...')
    ptr = 0
    sishu = 768
    gall_feat = np.zeros((ngall, sishu))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input,input,input,input,label=label)
            
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    ptr = 0
    query_feat = np.zeros((nquery, sishu))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input,input,input,input,label=label)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    if dataset == 'sysu':
        cmc, mAP, mInp = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'regdb':
        cmc, mAP, mInp = eval_regdb(distmat, query_label, gall_label)
    elif dataset == 'llcm':
        cmc, mAP, mInp = eval_llcm(distmat, query_label, gall_label, query_cam, gall_cam)
    return cmc, mAP, mInp


# Training
best_mAP = 0
print('==> Start Training...')
for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH + 1):

    print('==> Preparing Data Loader...')

    sampler_rgb = IdentitySampler(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label,
                                  color_pos_rgb,thermal_pos_rgb, cfg.BATCH_SIZE, per_img=cfg.NUM_POS)

    # RGB-IR
    trainset_rgb.cIndex = sampler_rgb.index1  # color index
    trainset_rgb.tIndex = sampler_rgb.index2



    if cfg.METHOD == 'PMT':
        if epoch <= cfg.PL_EPOCH:
            sampler_gray = IdentitySampler(trainset_gray.train_color_label, trainset_gray.train_thermal_label,
                                           color_pos_rgb, thermal_pos_gray, cfg.BATCH_SIZE, per_img=cfg.NUM_POS)  # Gray
            # Gray-IR
            trainset_gray.cIndex = sampler_gray.index1
            trainset_gray.tIndex = sampler_gray.index2
            trainloader = data.DataLoader(trainset_gray, batch_size=cfg.BATCH_SIZE, sampler=sampler_gray,
                                        num_workers=args.num_workers,drop_last=True, pin_memory=True)
        else:
            trainloader = data.DataLoader(trainset_rgb, batch_size=cfg.BATCH_SIZE, sampler=sampler_rgb,
                                    num_workers=args.num_workers, drop_last=True, pin_memory=True)
    else:
        trainloader = data.DataLoader(trainset_rgb, batch_size=cfg.BATCH_SIZE, sampler=sampler_rgb,
                                    num_workers=args.num_workers, drop_last=True, pin_memory=True)

    train(epoch)

    if epoch > args.start_test and epoch % args.test_epoch == 0:
        cmc, mAP, mINP = test(query_loader, gall_loader, cfg.DATASET)
        print(' mAP: {:.2%} | mInp:{:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(mAP,mINP,cmc[0],cmc[4],cmc[9],cmc[19]))

        if mAP >= best_mAP:
            best_mAP = mAP
            if cfg.DATASET == 'sysu':
                torch.save(model.state_dict(), osp.join('./save_model', os.path.basename(args.config_file)[:-4] + '_best.pth'))  # maybe not the best
            else:
                torch.save(model.state_dict(), osp.join('./save_model/', os.path.basename(args.config_file)[:-4] + '_best_trial_{}.pth'.format(args.trial)))
    if cfg.DATASET == 'regdb':
        if epoch > 48 and epoch % args.save_epoch == 0:

            torch.save(model.state_dict(), osp.join('./save_model/', os.path.basename(args.config_file)[:-4]  + '_epoch{}.pth'.format(epoch)))
    else:
        if epoch > 20 and epoch % args.save_epoch == 0:

            torch.save(model.state_dict(), osp.join('./save_model', os.path.basename(args.config_file)[:-4]  + '_epoch{}.pth'.format(epoch)))