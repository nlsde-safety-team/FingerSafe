from FingerprintDataset import FingerprintTest
import argparse
from fingernet import deploy
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def run():
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int,
                        help='batch size', default=6)
    parser.add_argument('--method', '-d', type=str,
                        help='root of loading data',
                        default="clean")
    args = parser.parse_args()

    pkl_path = './datapaths/datapath_MHS_{}_test.pkl'.format(args.method)
    f = open(pkl_path, 'rb')
    a = pickle.load(f)
    f.close()
    for k1 in sorted(a.keys()):
        for k2 in sorted(a[k1].keys()):
            print(a[k1][k2])
            mnt = deploy(a[k1][k2])
            np.save(a[k1][k2].replace('.bmp', '.npy'), mnt)

if __name__ == '__main__':
    run()
