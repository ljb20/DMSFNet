from __future__ import print_function
import torch
import ast
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from dataloader import TestData, TestData_RegDB
from datamanager import *
from model.make_model import build_vision_transformer, EDITOR
from config.config import cfg
from transforms import transform_test
from tqdm import tqdm
from NFCC import NFC
import argparse

parser = argparse.ArgumentParser(description='PMT Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu or llcm')
parser.add_argument('--config_file', default='config/SYSU.yml',
                    help='path to config file', type=str)
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--test-batch', default=64, type=int,
                    help='testing batch size')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None, nargs=argparse.REMAINDER)
parser.add_argument('--mode', default='all', type=str,
                    help='all or indoor for sysu')


def str_to_list(value):
    return ast.literal_eval(value)


parser.add_argument('--test_mode', default=[1, 2], type=str_to_list,
                    help='test mode for llcm')  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;
parser.add_argument('--gall_mode', default='single', type=str,
                    help='single or multi for sysu')
parser.add_argument('--trial', default=1, type=int, help='trial for regdb and LLCM')
parser.add_argument('--tvsearch', type=int, choices=[0, 1],
                    help='whether thermal to visible search on regdb')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.config_file != '':
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

if args.dataset == 'sysu':
    data_path = cfg.DATA_PATH_SYSU
    n_class = 395
elif args.dataset == 'regdb':
    data_path = cfg.DATA_PATH_RegDB
    args.tvsearch = bool(args.tvsearch)
    n_class = 206
elif args.dataset == 'llcm':
    data_path = cfg.DATA_PATH_LLCM
    n_class = 713
    test_mode = args.test_mode
    print(test_mode)

print('==> Building model..')
model = EDITOR(num_classes=n_class, cfg=cfg)
model.to(device)
cudnn.benchmark = True
model.eval()
sishu = 768


def extract_gall_feat(gall_loader):
    model.eval()
    ptr = 0
    gall_feat = np.zeros((ngall, sishu))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input, input, input, input, label=label)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    return gall_feat


def extract_query_feat(query_loader):
    model.eval()
    ptr = 0
    query_feat = np.zeros((nquery, sishu))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input, input, input, input, label=label)
            # feat = NFC(feat)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    return query_feat


all_cmc = []
all_mAP = []
all_mINP = []

if args.dataset == 'sysu':
    # load checkpoint
    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = args.model_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            model.load_param(model_path)
            print('==> loaded checkpoint {}'.format(args.resume))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # Test set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0,
                                                          gall_mode=args.gall_mode)  # indoor

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    query_loader = data.DataLoader(queryset, batch_size=128, shuffle=False, num_workers=args.workers)

    query_feat = extract_query_feat(query_loader)

    for i in tqdm(range(10)):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=i,
                                                              gall_mode=args.gall_mode)  # all

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False,
                                            num_workers=args.workers)

        gall_feat = extract_gall_feat(trial_gall_loader)

        distmat = -np.matmul(query_feat, np.transpose(gall_feat))

        cmc, mAP, mInp = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        print(
            '\n mAP: {:.2%} | mInp:{:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%} '.format(mAP,
                                                                                                                    mInp,
                                                                                                                    cmc[
                                                                                                                        0],
                                                                                                                    cmc[
                                                                                                                        4],
                                                                                                                    cmc[
                                                                                                                        9],
                                                                                                                    cmc[
                                                                                                                        19]))

        all_cmc.append(cmc)
        all_mAP.append(mAP)
        all_mINP.append(mInp)

    all_cmc = np.array(all_cmc)
    all_cmc1 = np.mean(all_cmc, axis=0)
    all_mAP1 = np.mean(all_mAP)
    all_mINP1 = np.mean(all_mINP)
    print('\n Average:')
    print('mAP: {:.2%} | mInp:{:.2%} | top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(all_mAP1,
                                                                                                              all_mINP1,
                                                                                                              all_cmc1[
                                                                                                                  0],
                                                                                                              all_cmc1[
                                                                                                                  4],
                                                                                                              all_cmc1[
                                                                                                                  9],
                                                                                                              all_cmc1[
                                                                                                                  19]))

all_cmc2 = []
all_mAP2 = []
all_mINP2 = []

if args.dataset == 'llcm':
    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = args.model_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            model.load_param(model_path)
            print('==> loaded checkpoint {}'.format(args.resume))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))
    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    query_feat = extract_query_feat(query_loader)
    for trial in tqdm(range(10)):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=trial)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False,
                                            num_workers=args.workers)
        gall_feat = extract_gall_feat(trial_gall_loader)

        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        cm, mA, mInp = eval_llcm(distmat, query_label, gall_label, query_cam, gall_cam)
        print(
            'mAP: {:.2%} | mInp:{:.2%} | R-1: {:.2%} | R-5: {:.2%} | R-10: {:.2%}| R-20: {:.2%}'.format(mA, mInp, cm[0],
                                                                                                        cm[4], cm[9],
                                                                                                        cm[19]))
        all_cmc2.append(cm)
        all_mAP2.append(mA)
        all_mINP2.append(mInp)

    all_cmc2 = np.array(all_cmc2)
    cm = np.mean(all_cmc2, axis=0)
    mA = np.mean(all_mAP2)
    mInp = np.mean(all_mINP2)
    print('\n Average:')
    print('mAP: {:.2%} | mInp:{:.2%} | R-1: {:.2%} | R-5: {:.2%} | R-10: {:.2%}| R-20: {:.2%}'.format(mA, mInp, cm[0],
                                                                                                      cm[4], cm[9],
                                                                                                      cm[19]))

all_cmc3 = []
all_mAP3 = []
all_mINP3 = []

if args.dataset == 'regdb':
    # load checkpoint
    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = args.model_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            model.load_param(model_path)
            print('==> loaded checkpoint {}'.format(args.resume))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    for trial in tqdm(range(1, 11)):
        query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
        queryset = TestData_RegDB(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        query_feat = extract_query_feat(query_loader)
        gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')
        galleryset = TestData_RegDB(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        gall_loader = data.DataLoader(galleryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        gall_feat = extract_gall_feat(gall_loader)

        if args.tvsearch:  # T -> V
            distmat = -np.matmul(gall_feat, np.transpose(query_feat))
            cm, mA, mInp = eval_regdb(distmat, gall_label, query_label)
            all_cmc3.append(cm)
            all_mAP3.append(mA)
            all_mINP3.append(mInp)
            print('mAP: {:.2%} | mInp:{:.2%} | R-1: {:.2%} | R-5: {:.2%} | R-10: {:.2%}| R-20: {:.2%}'.format(mA, mInp,
                                                                                                              cm[0],
                                                                                                              cm[4],
                                                                                                              cm[9],
                                                                                                              cm[19]))
        else:  # V -> T
            distmat = -np.matmul(query_feat, np.transpose(gall_feat))
            cm, mA, mInp = eval_regdb(distmat, query_label, gall_label)
            all_cmc3.append(cm)
            all_mAP3.append(mA)
            all_mINP3.append(mInp)
            print('mAP: {:.2%} | mInp:{:.2%} | R-1: {:.2%} | R-5: {:.2%} | R-10: {:.2%}| R-20: {:.2%}'.format(mA, mInp,
                                                                                                              cm[0],
                                                                                                              cm[4],
                                                                                                              cm[9],
                                                                                                              cm[19]))

    all_cmc3 = np.array(all_cmc3)
    cm = np.mean(all_cmc3, axis=0)
    mA = np.mean(all_mAP3)
    mInp = np.mean(all_mINP3)
    print('\n Average:')
    print('mAP: {:.2%} | mInp:{:.2%} | R-1: {:.2%} | R-5: {:.2%} | R-10: {:.2%}| R-20: {:.2%}'.format(mA, mInp, cm[0],
                                                                                                      cm[4], cm[9],
                                                                                                      cm[19]))