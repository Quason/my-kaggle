import os
import random
import cv2
import numpy as np
import shutil
from os.path import join, split
from glob import glob
from tqdm import tqdm


IMG_TYPE_LUT = {
    'large_bowel': 1,
    'small_bowel': 2,
    'stomach': 3
}


def rle_decoder(img_dir, csv_fn, dst_dir):
    # case2_day4_slice_0045
    # case2/case2_day1/scans/slice_0001_266_266_1.50_1.50.png
    with open(csv_fn, 'r') as fp:
        lines = fp.readlines()
        for line in tqdm(lines[1:]):
            line = line.replace('\n', '')
            line_item = line.split(',')
            line_item = [item for item in line_item if item]
            name = line_item[0]
            img_class = IMG_TYPE_LUT[line_item[1]]
            if len(line_item) == 3:
                rle_code = line_item[-1]
                rle_code = rle_code.split(' ')
                rle_code = np.array([int(i) for i in rle_code])
                rle_code = rle_code.reshape((-1, 2))
            else:
                rle_code = None
            case_name = name.split('_')[0]
            day_name = name.split('_')[1]
            slice_name = name.split('_')[-1]
            img_fn = glob(join(
                img_dir,
                case_name,
                f'{case_name}_{day_name}',
                'scans',
                f'slice_{slice_name}_*'
            ))
            img_fn = img_fn[0]
            img_data = cv2.imread(img_fn, -1)
            mask_data = np.zeros(img_data.size, np.uint8)
            if rle_code is not None:
                for i in range(rle_code.shape[0]):
                    i0 = rle_code[i][0]
                    i1 = rle_code[i][0] + rle_code[i][1]
                    mask_data[i0:i1] = img_class
            mask_data = mask_data.reshape(img_data.shape)
            dst_dir_tmp = join(dst_dir, f'{case_name}_{day_name}')
            os.makedirs(dst_dir_tmp, exist_ok=True)
            mask_fn = join(dst_dir_tmp, f'{case_name}_{day_name}_{slice_name}_mask.png')
            img_fn_cp = join(dst_dir_tmp, f'{case_name}_{day_name}_{slice_name}.png')
            shutil.copy(img_fn, img_fn_cp)
            if os.path.exists(mask_fn):
                mask_data_0 = cv2.imread(mask_fn, -1)
                fg_key = mask_data_0 != 0
                mask_data[fg_key] = 0
                mask_data += mask_data_0
            cv2.imwrite(mask_fn, mask_data)


def ds_split(src_dir, target_dir, blank_dir):
    ''' 将有分割目标和无分割目标的分开存放（方便后面双阶段模型训练）
        统计大小: 266: 60%; 276: 5%; 360: 35%
    '''
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(blank_dir, exist_ok=True)
    # import matplotlib.pyplot as plt
    fns = glob(join(src_dir, '*_mask.png'))
    sub_dirs = glob(join(src_dir, '*'))
    for item in sub_dirs:
        fns += glob(join(item, '*_mask.png'))
    size_stat = {}
    for fn in tqdm(fns):
        mask_data = cv2.imread(fn, -1)
        img_fn = fn.replace('_mask.png', '.png')
        img_data = cv2.imread(img_fn, -1)
        if np.max(mask_data) == 0:
            cv2.imwrite(join(blank_dir, split(img_fn)[-1]), img_data)
            cv2.imwrite(join(blank_dir, split(fn)[-1]), mask_data)
        else:
            cv2.imwrite(join(target_dir, split(img_fn)[-1]), img_data)
            cv2.imwrite(join(target_dir, split(fn)[-1]), mask_data)
        img_size = np.max(mask_data.shape)
        if img_size in size_stat:
            size_stat[img_size] += 1
        else:
            size_stat[img_size] = 1
    plot_bar = []
    for key in size_stat:
        plot_bar.append([key, size_stat[key]])
    plot_bar = np.array(plot_bar)
    # plt.bar(plot_bar[:, 0], plot_bar[:, 1])
    # plt.show()


def train_test_split(input_dir, output_dir_train, output_dir_test, train_ratio=0.7):
    ''' train and test dataset split
    '''
    mask_fns = glob(join(input_dir, '*mask.png'))
    sub_dirs = glob(join(input_dir, '*'))
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)
    for sub_dir in sub_dirs:
        if os.path.isdir(sub_dir):
            tmp_fns = glob(join(sub_dir, '*mask.png'))
            mask_fns += tmp_fns
    fn_cnt = len(mask_fns)
    fn_index = [i for i in range(fn_cnt)]
    random.shuffle(fn_index)
    for i in tqdm(range(fn_cnt)):
        mask_fn = mask_fns[fn_index[i]]
        img_fn = mask_fn.replace('_mask.png', '.png')
        if i < (fn_cnt * train_ratio):
            mask_fn_cp = join(output_dir_train, split(mask_fn)[-1])
            img_fn_cp = join(output_dir_train, split(img_fn)[-1])
        else:
            mask_fn_cp = join(output_dir_test, split(mask_fn)[-1])
            img_fn_cp = join(output_dir_test, split(img_fn)[-1])
        shutil.copy(mask_fn, mask_fn_cp)
        shutil.copy(img_fn, img_fn_cp)


if __name__ == '__main__':
    rle_decoder(
        '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/train',
        '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/train.csv',
        '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/ds'
    )

    ds_split(
        '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/ds',
        '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/ds_has_target/all',
        '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/ds_no_target/all'
    )

    train_test_split(
        '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/ds_no_target/all',
        '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/ds_no_target/train',
        '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/ds_no_target/test',
        train_ratio=0.7
    )
