import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import join


def main(src_dir):
    fns = glob(join(src_dir, '*mask.png'))
    sub_dirs = glob(join(src_dir, '*'))
    for sub_dir in sub_dirs:
        if os.path.isdir(sub_dir):
            fns += glob(join(sub_dir, '*mask.png'))
    fns = random.sample(fns, 10000)
    img_max = []
    for fn in tqdm(fns):
        img_fn = fn.replace('_mask.png', '.png')
        data = cv2.imread(img_fn, -1)
        img_max.append(np.max(data))
    return img_max


if __name__ == '__main__':
    main()
