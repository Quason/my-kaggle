import os
import cv2
import random
import numpy as np
import albumentations as A
from os.path import join, split
from torch.utils.data import Dataset
from glob import glob
from gi_tract_seg_2022.config import IMG_AUG


class GIDataset(Dataset):
    ''' 数据集统计特征
        height: 18 / 152 / 56
        width: 15 / 123 / 54
        size: 324 / 15525 / 3276
        resize: 64*64
    '''
    def __init__(self, src_dir, return_name=False, random_aug=False,
                 mixup=False, target_shape=(256,256), sample_ratio=None):
        ''' dataset init
            src_dir: input path
            return_name: whether to return base name
            random_aug: whether to do random augmentation
            class_equal: whether to do sample balance
            mixup
        '''
        super().__init__()
        fns = glob(join(src_dir, '*mask.png'))
        sub_dirs = glob(join(src_dir, '*'))
        for sub_dir in sub_dirs:
            if os.path.isdir(sub_dir):
                fns += glob(join(sub_dir, '*mask.png'))
        self.return_name = return_name
        self.random_aug = random_aug
        self.mixup = mixup
        self.target_shape = target_shape
        if sample_ratio is not None:
            fns = random.sample(fns, int(len(fns)*0.5))
        self.fns = fns

    def _img_aug(self, img_data, mask_data):
        seq = A.Compose(
            [
                A.HorizontalFlip(p=IMG_AUG['p_hflip']),  # 水平翻转
                A.VerticalFlip(p=IMG_AUG['p_vflip']),    # 垂直翻转
                A.RandomRotate90(p=IMG_AUG['p_rotate90']), # 旋转
                A.Transpose(p=IMG_AUG['p_transpose']),
            ],
            additional_targets={'img_mask': 'image'}
        )
        aug_res = seq(image=img_data, img_mask=mask_data)
        img_aug = aug_res['image']
        mask_aug = aug_res['img_mask']
        return img_aug, mask_aug

    def __getitem__(self, index):
        mask_data = cv2.imread(self.fns[index], -1)
        img_fn = self.fns[index].replace('_mask.png', '.png')
        img_data = cv2.imread(img_fn, -1)
        mask_data = cv2.resize(mask_data, self.target_shape, cv2.INTER_NEAREST)
        img_data = cv2.resize(img_data, self.target_shape, cv2.INTER_LINEAR)
        if self.mixup:
            img_h, img_w, img_c = img_data.shape
            img_mixup = np.zeros((img_h*2, img_w*2, img_c), np.uint16)
            mask_mixup = np.zeros((img_h*2, img_w*2), np.uint8)
            for i in range(4):
                img_aug, mask_aug = self._img_aug(img_data, mask_data)
                ih = i // 2
                iw = i % 2
                img_mixup[ih*img_h:(ih+1)*img_h, iw*img_w:(iw+1)*img_w, :] = img_aug
                mask_mixup[ih*img_h:(ih+1)*img_h, iw*img_w:(iw+1)*img_w] = mask_aug
            img_data = img_mixup
            mask_data = mask_mixup
        else:
            if self.random_aug:
                img_data, mask_data = self._img_aug(img_data, mask_data)
        max_value = np.max(img_data)
        img_data = img_data / max_value  # normalization
        img_data[img_data > 1] = 1
        img_data = np.expand_dims(img_data, axis=0)
        if self.return_name:
            base_name = split(img_fn)[-1]
            return img_data, mask_data, base_name
        else:
            return img_data, mask_data
    
    def __len__(self):
        return len(self.fns)
