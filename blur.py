# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
from PIL import Image
from scipy import signal
import cv2
import random

import glob, os


class GaussianBlur(object):
    def __init__(self, kernel_size=3, sigma=1.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.gaussian_kernel()

    def gaussian_kernel(self):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size), dtype=float)
        radius = self.kernel_size // 2
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(-1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v
        kernel2 = kernel / np.sum(kernel)
        return kernel2

    # def filter(self, img: Image.Image):
    def filter(self, img):
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            new_arr = signal.convolve2d(img_arr, self.kernel, mode="same", boundary="symm")
        else:
            h, w, c = img_arr.shape
            new_arr = np.zeros(shape=(h, w, c), dtype=float)
            for i in range(c):
                new_arr[..., i] = signal.convolve2d(img_arr[..., i], self.kernel, mode="same", boundary="symm")
        new_arr = np.array(new_arr, dtype=np.uint8)
        return Image.fromarray(new_arr)
        return new_arr

def mask_blur(root_dir, filename, root_mask, mask_file, mykernel, mysigma):
    img = cv2.imread(os.path.join(root_dir, filename))
    img = cv2.resize(img, (224, 224))
    mask = cv2.imread(os.path.join(root_mask, mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (224, 224))
    # mask = mask.astype(np.bool_)
    # img_mask = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    img_mask = img

    W, H, _ = np.shape(img)
    for w in range(W):
        for h in range(H):
            if mask[w][h] == 0:
                img_mask[w, h, :] = 0
    blurred = GaussianBlur(kernel_size=mykernel, sigma=mysigma).filter(img_mask)
    return blurred

def random_blur(root_dir, filename, mykernel, mysigma):
    img = cv2.imread(os.path.join(root_dir, filename))
    mask = np.array(img)
    W, H, _ = np.shape(img)
    size = 150  # PolyU: 150=30%, 200=50%
    left = random.randint(0, W - size)
    right = random.randint(0, H - size)
    blurred = np.zeros((size, size, 3))
    # left = 650
    # right = 250
    for w in range(left, left + size):
        for h in range(right, right + size):
            blurred[w - left][h - right] = img[w][h]
    blurred = GaussianBlur(kernel_size=mykernel, sigma=mysigma).filter(blurred)

    for w in range(W):
        for h in range(H):
            mask[w][h] = 0
    for w in range(left, left + size):
        for h in range(right, right + size):
            mask[w][h] = 255
            img[w][h] = blurred[w - left][h - right]
    return img, mask


def random_mosaic(root_dir, filename, neighbor=4):
    img = cv2.imread(os.path.join(root_dir, filename))
    mask = np.array(img)
    W, H, _ = np.shape(img)
    size = 150  # PolyU: 150=30%, 200=50%
    left = random.randint(0, W - size)
    right = random.randint(0, H - size)
    # left = 650
    # right = 350
    blurred = np.zeros((size, size, 3))
    for w in range(left, left + size):
        for h in range(right, right + size):
            blurred[w - left][h - right] = img[w][h]
    # blurred = GaussianBlur(kernel_size=mykernel, sigma=mysigma).filter(blurred)
    mosaic = do_mosaic(blurred, 0, 0, size, size, neighbor=neighbor)

    for w in range(W):
        for h in range(H):
            mask[w][h] = 0
    for w in range(left, left + size):
        for h in range(right, right + size):
            mask[w][h] = 255
            img[w][h] = mosaic[w - left][h - right]
    return img, mask


def mask_random_mosaic(root_dir, filename, position_dir, position_name, neighbor=4):
    img = cv2.imread(os.path.join(root_dir, filename))
    position = cv2.imread(os.path.join(position_dir, position_name), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    position = cv2.resize(position, (224, 224))

    mask = np.array(img)
    W, H, _ = np.shape(img)
    size = 110
    left, right = W - size, 0
    for h in range(H):
        if position[W-1, h] == 255:
            right = h
            break
    blurred = np.zeros((size, size, 3))
    for w in range(W - size, W):
        for h in range(right, right + size):
            blurred[w-left, h - right, :] = img[w, h, :]
    # blurred = GaussianBlur(kernel_size=mykernel, sigma=mysigma).filter(blurred)
    mosaic = do_mosaic(blurred, 0, 0, size, size, neighbor=neighbor)

    for w in range(W):
        for h in range(H):
            mask[w][h] = 0
    for w in range(left, left + size):
        for h in range(right, right + size):
            mask[w][h] = 255
            img[w][h] = mosaic[w - left][h - right]
    return img, mask

def mask_mosaic(root_dir, filename, position_dir, position_name, neighbor=4):
    img = cv2.imread(os.path.join(root_dir, filename))
    position = cv2.imread(os.path.join(position_dir, position_name), cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (224, 224))
    # position = cv2.resize(position, (224, 224))

    W, H, _ = np.shape(img)
    size = 175  # PolyU: 150=30%, 200=50%
    left, right = W - size, 0

    for w in range(W):
        for h in range(H):
            if position[w][h] == 255:
                left = w
                right = h
                break
    left = left - size
    blurred = np.zeros((size, size, 3))
    for w in range(left, left + size):
        for h in range(right, right + size):
            blurred[w - left, h - right, :] = img[w, h, :]
    mosaic = do_mosaic(blurred, 0, 0, size, size, neighbor=neighbor)

    for w in range(left, left + size):
        for h in range(right, right + size):
            img[w][h] = mosaic[w - left][h - right]
    return img

def do_mosaic(img, x, y, w, h, neighbor=4):
    """
    :param rgb_img
    :param int x :  马赛克左顶点
    :param int y:  马赛克左顶点
    :param int w:  马赛克宽
    :param int h:  马赛克高
    :param int neighbor:  马赛克每一块的宽
    """
    for i in range(0, h, neighbor):
        for j in range(0, w, neighbor):
            rect = [j + x, i + y]
            color = img[i + y][j + x].tolist()
            left_up = (rect[0], rect[1])
            x2 = rect[0] + neighbor - 1
            y2 = rect[1] + neighbor - 1
            if x2 > x + w:
                x2 = x + w
            if y2 > y + h:
                y2 = y + h
            right_down = (x2, y2)
            cv2.rectangle(img, left_up, right_down, color, -1)

    return img

import os
if __name__ == '__main__':
    mykernel = 15
    mysigma = 2.5
    root_dir = './datasets/final/test'
    position_dir = './datasets/final/blur_position_2'
    blurred_dir = 'datasets/final/blur_2'
    if not os.path.exists(position_dir):
        os.mkdir(position_dir)
    if not os.path.exists(blurred_dir):
        os.mkdir(blurred_dir)

    img = mask_mosaic('./datasets/final/test', '331_2.bmp', './datasets/final/mosac_position_40', '331_3.bmp', neighbor=20)
    cv2.imwrite('./segment/331_40.jpg', img)
    exit(0)

    # mask_dir = './datasets/physical_square/mosac_50_position'

    # for classname in os.listdir(root_dir):
    #     class_dir = os.path.join(root_dir, classname)
    #     position_class = os.path.join(position_dir, classname)
    #     mask_class = os.path.join(mask_dir, classname)
    #     blur_class = os.path.join(blurred_dir, classname)
    #
    #     if not os.path.exists(blur_class):
    #         os.makedirs(blur_class)
    #     if not os.path.exists(mask_class):
    #         os.makedirs(mask_class)

    # for filename in os.listdir(root_dir):
    #     img = cv2.imread(os.path.join(root_dir, filename))
    #     img2 = GaussianBlur(kernel_size=mykernel, sigma=mysigma).filter(img)
    #     # blurred, mask = random_blur(root_dir, filename, mykernel, mysigma)
    #     # mosaic_img, mask = random_mosaic(root_dir, filename, neighbor=10)
    #     # blurred = mask_blur(class_dir, filename, mask_class, filename, mykernel, mysigma)
    #     # mosaic_img, mask = mask_random_mosaic(class_dir, filename, position_class, filename, neighbor=10)
    #     cv2.imwrite(os.path.join(blurred_dir, filename), img2)
    #     # cv2.imwrite(os.path.join(position_dir, filename), mask)
    #     print(filename)
    # exit(0)

    # for class_name in os.listdir(root_dir):
    #     class_dir = os.path.join(root_dir, class_name)
    #     mask_class = os.path.join(mask_dir, class_name)
    #     blur_class = os.path.join(blurred_dir, class_name)
    #     if not os.path.exists(blur_class):
    #         os.makedirs(blur_class)
    #     for filename in os.listdir(class_dir):
    #         blurred, mask = random_blur(root_dir, filename, mykernel, mysigma)
    #         # mosaic_img, mask = random_mosaic(root_dir, filename, neighbor=10)
    #         # blurred = mask_blur(class_dir, filename, mask_class, filename, mykernel, mysigma)
    #         cv2.imwrite(os.path.join(blur_class, filename), blurred)
    #         # cv2.imwrite(os.path.join(mask_dir, filename), mask)
    #         print(filename)

    img = Image.open("./datasets/final/train/f1/p1.bmp").convert("RGB")
    H, W = img.size
    # img = img.resize((224, 325))
    img2 = GaussianBlur(kernel_size=mykernel, sigma=mysigma).filter(img)
    # img2 = img2.resize((H, W))
    img2.save("./segment/train_blur_2.jpg")
    exit(0)


    #批量处理
    root_dir = '[PATH]'
    for filename in glob.iglob(root_dir + '/*.bmp', recursive=False):

        img = Image.open(filename).convert("RGB")
        img2 = GaussianBlur(kernel_size=mykernel, sigma=mysigma).filter(img)

        base = os.path.basename(filename)

        outfilename = os.path.dirname(root_dir) + '/blurred25/' + base
        img2.save(outfilename)

        #print("【Done】" + outfilename)
        pass
