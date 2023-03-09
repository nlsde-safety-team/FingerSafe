import cv2
import os
from skimage.metrics import structural_similarity as ssim
import torch
from FingerprintDataset import FingerprintTest

# root = './datasets/fingerprint_colored/f1'
# pa = cv2.imread(os.path.join(root, 'train/p1.bmp'))
# pa = cv2.resize(pa, (224, 224))
# pb = cv2.imread(os.path.join(root, 'perturb_nopre/p1.bmp'))
# print(ssim(pa, pb, multichannel=True))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataroot = './datasets/final/train'
    raw_set = FingerprintTest(dataroot, '')
    adv_set = FingerprintTest('./datasets/final/perturb_LC', '')
    total = 0
    for i in range(len(raw_set.y_data)):
        pa = raw_set.x_data[i].permute(1, 2, 0).numpy()
        pb = adv_set.x_data[i].permute(1, 2, 0).numpy()
        # total += ssim(pa, pb, multichannel=True)
        total += psnr2(pa, pb)
    SSIM = total / len(raw_set)
    print(SSIM)


import imageio
import numpy as np
import math


def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    main()
