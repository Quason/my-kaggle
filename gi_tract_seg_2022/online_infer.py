# demo: case123_day20_slice_0001,large_bowel, 1 2 12 5
# /kaggle/train/uw-madison-gi-tract-image-segmentation/train/case119/case119_day0/scans
import os
import cv2
import torch
import numpy as np
from os.path import join
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from gi_tract_seg_2022.models import mscff


IMG_TYPE_LUT = {
    'large_bowel': 1,
    'small_bowel': 2,
    'stomach': 3
}


def rle_encoder(img):
    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 1
    ends_ix = np.where(ends)[0] + 1
    lengths = ends_ix - starts_ix
    return starts_ix, lengths


class GIDataset(Dataset):
    def __init__(self, src_dir, target_shape=(128,128)):
        super().__init__()
        fns = []
        sub_dirs = os.walk(src_dir)
        for item in tqdm(sub_dirs):
            dir_name, _, sub_files = item
            if len(sub_files) > 0:
                for fn in sub_files:
                    if fn[-4:] == '.png':
                        name = join(dir_name, fn)
                        case_day = dir_name.split('/')[-2]
                        slice_name = fn.split('_')[1]
                        name_uid = f'{case_day}_clice_{slice_name}'
                        fns.append([name, name_uid])
        self.fns = fns
        self.target_shape = target_shape

    def __getitem__(self, index):
        img_fn, name_uid = self.fns[index]
        img_data = cv2.imread(img_fn, -1)
        raw_size = img_data.shape
        img_data = cv2.resize(img_data, self.target_shape, cv2.INTER_LINEAR)
        max_value = np.max(img_data)
        img_data = img_data.astype(np.float32) / max_value  # normalization
        img_data[img_data > 1] = 1
        img_data = np.expand_dims(img_data, axis=0)
        return img_data, name_uid, raw_size

    def __len__(self):
        return len(self.fns)


def main():
    src_dir = '/data/qiyuan_data/uw-madison-gi-tract-image-segmentation/train'
    model_fn = '/root/qiyuan/codes/my-kaggle/data/tflogs/20230411071211/E478.pth'
    dst_fn = '/root/qiyuan/codes/my-kaggle/data/output_test/submission.csv'
    bsize = 20

    fp = open(dst_fn, 'w')
    fp.write('id,class,predicted\n')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pred_dataset = GIDataset(src_dir, target_shape=(128,128))
    pred_loader = DataLoader(pred_dataset, batch_size=bsize, shuffle=False)
    net = mscff.MSCFF(1, 4)
    net.load_state_dict(torch.load(model_fn, map_location=device))
    net.to(device=device)
    net.eval()
    for batch in tqdm(pred_loader):
        imgs, name_uid, raw_size = batch
        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_pred = torch.squeeze(torch.argmax(net(imgs), dim=1))
        mask_pred = mask_pred.cpu().detach().numpy()

        for bi in range(imgs.shape[0]):
            mask_pred_tmp = mask_pred[bi, :, :]
            raw_size_tmp = [raw_size[0][bi], raw_size[1][bi]]
            mask_pred_tmp = cv2.resize(
                mask_pred_tmp.astype(np.uint8),
                (int(raw_size_tmp[1]), int(raw_size_tmp[0])), cv2.INTER_NEAREST)
            for key in IMG_TYPE_LUT:
                mask_tmp = (mask_pred_tmp == IMG_TYPE_LUT[key]).astype(np.uint8)
                starts_ix, lengths = rle_encoder(mask_tmp)
                rle_code = []
                if len(starts_ix) > 0:
                    for i in range(len(starts_ix)):
                        rle_code.append(str(starts_ix[i]))
                        rle_code.append(str(lengths[i]))
                    rle_code = ' '.join(rle_code)
                else:
                    rle_code = ''
                fp.write(f'{name_uid[bi]},{key},{rle_code}\n')
    fp.close()


if __name__ == '__main__':
    main()
