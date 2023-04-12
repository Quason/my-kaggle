import torch
import pickle
from gi_tract_seg_2022.models import mscff


def trace():
    pth_fn = '/root/qiyuan/codes/my-kaggle/data/tflogs/20230411071211/E402.pth'
    pt_fn = '/root/qiyuan/codes/my-kaggle/data/tflogs/20230411071211/E402.pkl'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn_net = mscff.MSCFF(1, 4)
    cnn_net.to(device=device)
    cnn_net.load_state_dict(torch.load(pth_fn, map_location=device))
    with open(pt_fn, 'wb') as fp:
        pickle.dump(cnn_net, fp)
    fp.close()


if __name__ == '__main__':
    trace()
