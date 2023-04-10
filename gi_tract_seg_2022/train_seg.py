''' deep learning model train
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm
from os.path import join
from torch.utils.data import DataLoader
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

from gi_tract_seg_2022.models import mscff
from gi_tract_seg_2022.my_dataset import GIDataset
from gi_tract_seg_2022.config import LABEL_SMOOTHING


def metrics(mask, mask_pred):
    label_pred = torch.argmax(mask_pred, dim=1)
    intersect = (mask == label_pred) * (mask != 0) * (label_pred != 0)
    union = (mask != 0) + (label_pred != 0)
    miou = torch.sum(intersect) * 2 / torch.sum(union)
    oa = torch.sum(mask == label_pred) / mask.numel()
    return oa.cpu(), miou.cpu()


def train(train_dir, test_dir, model_type, save_dir,
          pretrain_pth=None, test_mixup=False,
          loss_weight=[0.1, 0.3, 0.3, 0.3],
          lr=1e-4,
          epochs=500,
          batch_size=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtime = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    tfdir = join(save_dir, dtime)
    os.makedirs(tfdir, exist_ok=True)
    writer = SummaryWriter(tfdir)
    input_channels = 1
    num_classes = len(loss_weight)

    if model_type == 'mscff':
        net = mscff.MSCFF(input_channels, num_classes)
    if pretrain_pth is not None:
        net.load_state_dict(torch.load(pretrain_pth))
    net.to(device=device)
    test_dataset = GIDataset(
        test_dir,
        return_name=False,
        random_aug=False,
        mixup=test_mixup,
        target_shape=(256,256)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(loss_weight), label_smoothing=LABEL_SMOOTHING
    ).to(device=device)

    miou_best = 0
    train_oa = []
    train_miou = []
    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        train_dataset = GIDataset(
            train_dir,
            return_name=True,
            random_aug=True,
            mixup=False,
            target_shape=(256,256),
            
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        # train dataset
        for batch in tqdm(train_loader):
            imgs, mask, name = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.long)
            mask_pred = net(imgs)
            loss = criterion(mask_pred, mask)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp_oa, tmp_miou = metrics(mask, mask_pred)
            train_oa.append(tmp_oa)
            train_miou.append(tmp_miou)

        print(f'epoch{epoch+1}:')
        train_oa_mean = np.mean(train_oa)
        train_miou_mean = np.mean(train_miou)
        loss_mean = np.mean(epoch_loss)
        print('loss=%.5f, train oa=%.5f' % (loss_mean, train_oa_mean))

        # test dataset
        net.eval()
        test_oa = []
        test_miou = []
        for batch in tqdm(test_loader):
            imgs, mask = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.long)
            mask_pred = net(imgs)
            tmp_oa, tmp_miou = metrics(mask, mask_pred)
            test_oa.append(tmp_oa)
            test_miou.append(tmp_miou)
        test_oa_mean = np.mean(test_oa)
        test_miou_mean = np.mean(test_miou)
        
        writer.add_scalar('loss', loss_mean, epoch+1)
        writer.add_scalar('oa/train', train_oa_mean, epoch+1)
        writer.add_scalar('miou/train', train_miou_mean, epoch+1)
        writer.add_scalar('oa/test', test_oa_mean, epoch+1)
        writer.add_scalar('miou/test', test_miou_mean, epoch+1)

        # save model
        print(f'miou_test: {test_miou_mean}')
        if (epoch >= 3) and (test_miou_mean > miou_best):
            miou_best = test_miou_mean
            torch.save(net.state_dict(), join(tfdir, 'E%03d.pth' % (epoch+1)))
