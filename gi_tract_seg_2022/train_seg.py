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
    intersect = torch.sum((mask == label_pred) * (mask != 0))
    union = torch.sum((mask != 0) + (label_pred != 0))
    return intersect.cpu(), union.cpu()


def train(train_dir, test_dir, model_type, save_dir,
          pretrain_pth=None, test_mixup=False,
          loss_weight=[0.1, 0.3, 0.3, 0.3],
          lr=1e-4,
          epochs=500,
          batch_size=10,
          sample_ratio=None,
          target_shape=None):
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
        target_shape=target_shape,
        sample_ratio=None
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss().to(device=device)

    miou_best = 0
    for epoch in range(epochs):
        print(f'epoch{epoch+1}:')
        net.train()
        epoch_loss = []
        train_dataset = GIDataset(
            train_dir,
            return_name=False,
            random_aug=True,
            mixup=False,
            target_shape=target_shape,
            sample_ratio=sample_ratio
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        # train dataset
        train_intersect = []
        train_union = []
        for batch in tqdm(train_loader):
            imgs, mask = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.long)
            mask_pred = net(imgs)
            loss = criterion(mask_pred, mask)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp_intersect, tmp_union = metrics(mask, mask_pred)
            train_intersect.append(tmp_intersect)
            train_union.append(tmp_union)

        train_miou = np.sum(train_intersect) / np.sum(train_union)
        loss_mean = np.mean(epoch_loss)
        print('train: loss=%.5f, miou=%.5f' % (loss_mean, train_miou))

        # test dataset
        net.eval()
        test_intersect = []
        test_union = []
        for batch in tqdm(test_loader):
            imgs, mask = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.long)
            mask_pred = net(imgs)
            tmp_intersect, tmp_union = metrics(mask, mask_pred)
            test_intersect.append(tmp_intersect)
            test_union.append(tmp_union)
        test_miou = np.sum(test_intersect) / np.sum(test_union)
        
        writer.add_scalar('loss', loss_mean, epoch+1)
        writer.add_scalar('miou/train', train_miou, epoch+1)
        writer.add_scalar('miou/test', test_miou, epoch+1)

        # save model
        print(f'miou_test: {test_miou}')
        if (epoch >= 3) and (test_miou > miou_best):
            miou_best = test_miou
            torch.save(net.state_dict(), join(tfdir, 'E%03d.pth' % (epoch+1)))
