# -*- coding: utf-8 -*-

import math
from aug_operation import Augment
from network import Restormer_Encoder, Restormer_Decoder, LongScaleFeatureExtraction, ShortScaleFeatureExtraction
from utils.dataset import H5Dataset
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia
import matplotlib.pyplot as plt
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'CrossFuse'

# Hyper-parameters for training
num_epochs = 160 # total epoches
epoch_gap = 40  # epoches of Phase I
lr = 1e-4
weight_decay = 0
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']

# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.
coeff_ssl_base = 0.1

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(LongScaleFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(ShortScaleFeatureExtraction(num_layers=1)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')


trainloader = DataLoader(H5Dataset('./dataset/RSM3.h5'),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

internal_aug = Augment()  # weak-aggr data augmentation

# weight adjustment
def calculate_beta(beta_base, current_step, total_steps):
    # 计算余弦衰减值
    cosine_decay = math.cos(math.pi * current_step / total_steps)
    # 调整范围到[0, 1]
    adjusted_decay = (cosine_decay + 1) / 2
    # 计算调整后的 beta 值
    beta = beta_base * adjusted_decay
    return beta

# 定义两个阶段的损失列表
losses_phase1 = []
losses_phase2 = []
for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < epoch_gap: #Phase I
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
        else:  #Phase II
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)

            ## 对融合结果进行强弱增强的自监督学习
            weak_fuse, aggr_fuse = internal_aug(data_Fuse)

            mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)
            # SSL损失
            coeff_ssl = calculate_beta(coeff_ssl_base, epoch, num_epochs-epoch_gap)
            ssl_loss = coeff_ssl * MSELoss(weak_fuse, aggr_fuse)
            loss = fusionloss + coeff_decomp * loss_decomp + ssl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )
    if epoch < epoch_gap:
        # 记录当前 epoch 的损失值
        losses_phase1.append(np.mean(loss.item()))
    else:
        losses_phase2.append(np.mean(loss.item()))

    # adjust the learning rate
    scheduler1.step()  
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6

    # 创建一个具有两个子图的图布
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # 绘制阶段1的损失曲线（前40个epoch）
    axes[0].plot(range(1, len(losses_phase1) + 1), losses_phase1, label='Phase I Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Phase I Training Loss')

    # 绘制阶段2的损失曲线（后80个epoch）
    axes[1].plot(range(1, len(losses_phase2) + 1), losses_phase2, label='Phase II Loss', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Phase II Training Loss')

    # 调整布局
    plt.tight_layout()

    # 保存图像到文件
    plt.savefig('./models/loss_curve.png')


if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/CrossFuse_" + "RSM3" + '.pth'))


