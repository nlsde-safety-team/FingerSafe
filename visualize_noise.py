import numpy as np
import os
import sys
import cv2
import numpy as np
import fingerprint_enhancer


def get_corr(figa, figb):
    meana = np.mean(figa)
    meanb = np.mean(figb)
    if len(figa.shape) == 3 and len(figb.shape) == 3:
        corr = np.sum(np.einsum('ijk, ijk -> ijk', (figa - meana), (figb - meanb))) / np.sqrt(
            np.sum(np.einsum('ijk, ijk -> ijk', (figa - meana), (figa - meana))) *
            np.sum(np.einsum('ijk, ijk -> ijk', (figb - meanb), (figb - meanb))))
    elif len(figa.shape) == 2 and len(figb.shape) == 3:
        corr = np.sum(np.einsum('ij, ijk -> ijk', (figa - meana), (figb - meanb))) / np.sqrt(
            np.sum(np.einsum('ij, ij -> ij', (figa - meana), (figa - meana))) *
            np.sum(np.einsum('ijk, ijk -> ijk', (figb - meanb), (figb - meanb))))
    elif len(figa.shape) == 3 and len(figb.shape) == 2:
        corr = np.sum(np.einsum('ijk, ij -> ijk', (figa - meana), (figb - meanb))) / np.sqrt(
            np.sum(np.einsum('ijk, ijk -> ijk', (figa - meana), (figa - meana))) *
            np.sum(np.einsum('ij, ij -> ij', (figb - meanb), (figb - meanb))))
    elif len(figa.shape) == 2 and len(figb.shape) == 2:
        corr = np.sum(np.einsum('ij, ij -> ij', (figa - meana), (figb - meanb))) / np.sqrt(
            np.sum(np.einsum('ij, ij -> ij', (figa - meana), (figa - meana))) *
            np.sum(np.einsum('ij, ij -> ij', (figb - meanb), (figb - meanb))))
    else:
        raise TypeError("dimension of two matrix should be either 2 or 3")
    return corr


def Frangi_Filter(img):
    from frangi_filter import FrangiFilter2D
    outIm = FrangiFilter2D(img)
    img = outIm * 10000*255
    return img


def IJCB2015(image):  # MHS
    img_medianBlur = cv2.medianBlur(image, 3)
    cv2.imshow("0", img_medianBlur)
    cv2.waitKey(0)
    img_His = cv2.equalizeHist(img_medianBlur)
    cv2.imshow("1", img_His)
    cv2.waitKey(0)
    img_GaussBlur = cv2.GaussianBlur(img_His, (9, 9), 2)
    img_sharpen = img_His - img_GaussBlur
    cv2.imshow("2", img_sharpen)
    cv2.waitKey(0)
    return img_sharpen


def IJCB2017(img):  # HG
    img_His = cv2.equalizeHist(img)
    out = fingerprint_enhancer.enhance_Fingerprint(img_His, ridge_filter_thresh=-1)
    return out


TARGET_IMG_SIZE = 224

image_pgd = cv2.imread('./datasets/exp_pics/final_flower_fingersafe_e8.jpg')
# image_pgd = cv2.resize(image_pgd, dsize=(375, 240)).astype('int')
w, h, _ = image_pgd.shape

image_clean = cv2.imread('./datasets/exp_pics/raw/f1/flower.jpg')
W, H, _ = image_clean.shape
image_pgd = image_pgd[w-W:w, :H, :]

# image_clean = cv2.resize(image_clean, dsize=(375, 240)).astype('int')

image_gray = cv2.imread('segment/new4.jpg', flags=cv2.IMREAD_GRAYSCALE)
# image_gray = cv2.resize(image_gray, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))#.astype('int')
# img_m = cv2.imread('./segment/0.15m_0_mask.jpg', flags=cv2.IMREAD_GRAYSCALE)
image_mask = IJCB2015(image_gray)

image_noise = np.abs(image_pgd - image_clean)
image_noise = (image_noise / 8 * 255)

noise = np.abs(image_clean.astype('uint8') - image_pgd.astype('uint8'))  # noise
mask = cv2.imread('./datasets/exp_pics/mask/f1/mask_flower.jpg', cv2.IMREAD_GRAYSCALE)
W, H, _ = np.shape(noise)
for w in range(W):
    for h in range(H):
        if mask[w][h] == 0:
            noise[w, h, :] = 0

cv2.imwrite('./result/clean.bmp', image_clean.astype('uint8'))
cv2.imwrite('./result/adv.bmp', image_pgd.astype('uint8'))
# cv2.imwrite('./result/noise.bmp', np.abs(image_clean.astype('uint8') - image_pgd.astype('uint8')))
# cv2.imwrite('./result/noise.bmp', image_noise)
cv2.imwrite('./result/noise.bmp', noise)
cv2.imwrite('./result/mask.bmp', image_mask.astype('uint8'))
# cv2.imwrite('./result/noise_pos.bmp', image_noise_pos.astype('uint8'))
# cv2.imwrite('./result/noise_neg.bmp', image_noise_neg.astype('uint8'))
# cv2.imwrite('./result/another.bmp', image_another.astype('uint8'))
# cv2.imwrite('./result/new.bmp', image_new.astype('uint8'))
# print(get_corr(image_noise, image_noise_pos))


cv2.waitKey(0)


