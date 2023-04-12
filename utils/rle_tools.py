import numpy as np


def rle_encoder(img):
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 1
    ends_ix = np.where(ends)[0] + 1
    lengths = ends_ix - starts_ix
    
    return starts_ix, lengths


if __name__ == '__main__':
    a = np.random.randint(0,10,(10,10))
    a[0, :] = 0
    a[:, 0] = 0
    a[-1, :] = 0
    a[:, -1] = 0
    a = (a>5).astype(np.uint8)
    a *= 0
    print(a)
    starts_ix, lengths = rle_encoder(a)
    print(starts_ix)
    print(lengths)
