from yacs.config import CfgNode as CN
cfg = CN()
cfg.SEED = 0

cfg.HEAD_KEEP=1
cfg.FREQUENCY_KEEP=18
cfg.AL=1
cfg.guangpu=0
# dataset
cfg.DATASET = 'sysu'    # sysu or regdb
cfg.DATA_PATH_SYSU = '/share/home/u2315363110/EOT/SYSU-MM01/'
cfg.DATA_PATH_RegDB = '/share/home/u2315363110/EOT/RegDB/'
cfg.DATA_PATH_LLCM = '/share/home/u2315363110/EOT/LLCM/'
cfg.PRETRAIN_PATH = '/share/home/u2315363110/EOT/jx_vit_base_p16_224-80ecf9dd.pth'


cfg.START_EPOCH = 1
cfg.MAX_EPOCH = 24
cfg.CLS_TOKEN_NUM = 1
cfg.H = 256
cfg.W = 128
cfg.BATCH_SIZE = 32  # num of images for each modality in a mini batch
cfg.NUM_POS = 4      #代表每个样本中所选取的正样本数量

# PMT
cfg.METHOD ='PMT'
cfg.PL_EPOCH = 6    # for PL strategy的迭代轮数
cfg.MSEL = 0.5      # weight for MSEL
cfg.DCL = 0.5       # weight for DCL
cfg.MARGIN = 0.1    # margin for triplet
cfg.cmsm = 1.00
cfg.tri2 = 0.05
cfg.tri3 = 0.05
cfg.k1 = 3
cfg.k2 = 3
# model
cfg.STRIDE_SIZE = [12,12]
cfg.DROP_OUT = 0.03
cfg.ATT_DROP_RATE = 0.0
cfg.DROP_PATH = 0.1

# optimizer
cfg.OPTIMIZER_NAME = 'AdamW'  # AdamW or SGD
cfg.MOMENTUM = 0.9    # for SGD

cfg.BASE_LR = 3e-4 #这是基础学习率（base learning rate），用于控制每次参数更新的步长大小。具体数值为3e-4，即0.0003
cfg.WEIGHT_DECAY = 1e-4 #这是权重衰减（weight decay）系数，用于控制模型参数在更新过程中的正则化程度。具体数值为1e-4，即0.0001
cfg.WEIGHT_DECAY_BIAS = 1e-4 #这是针对偏置项（bias）的权重衰减系数，用于控制更新偏置项时的正则化程度。具体数值为1e-4，即0.0001
cfg.BIAS_LR_FACTOR = 1     #这是决定偏置项学习率相对于其他参数学习率的因子。如果值为1，则偏置项和其他参数具有相同的学习率；如果值大于1，则偏置项的学习率会略微增加；如果值小于1，则偏置项的学习率会略微降低。

cfg.LR_PRETRAIN = 0.5  #这是控制预训练模型学习率的因子，该参数的值为0.5，即预训练模型的学习率是当前模型学习率的一半。
cfg.LR_MIN = 0.01      #这是学习率的下限，即学习率在训练过程中不会降低到该下限以下
cfg.LR_INIT = 0.01     #这是学习率的初始值，即训练开始时的学习率大小
cfg.WARMUP_EPOCHS = 3  #这是设定前几个epoch使用较小的学习率进行热身训练的轮数。热身训练的目的是避免模型出现过拟合或陷入局部最优解