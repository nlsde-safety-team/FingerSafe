import os
import torch
import cv2
# import Fingerprint_enhancer_our
import fingerprint_enhancer
import torchvision.transforms as transforms
from frangi_filter import FrangiFilter2D
from PIL import Image


def IJCB2015(image):
    img_medianBlur = cv2.medianBlur(image, 3)
    img_His = cv2.equalizeHist(img_medianBlur)
    img_GaussBlur = cv2.GaussianBlur(img_His, (9, 9), 2)
    img_sharpen = img_His - img_GaussBlur
    return img_sharpen


def IJCB2017(img):
    img_His = cv2.equalizeHist(img)
    out = fingerprint_enhancer.enhance_Fingerprint(img_His, ridge_filter_thresh=-1)
    return out


def Frangi_Filter(img):
    outIm = FrangiFilter2D(img)
    img = outIm * 10000
    return img

def preprocess(source_dir, target_dir):
    img_to_tensor = transforms.ToTensor()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    classes = os.listdir(source_dir)
    for iclass in classes:
        num_subject = iclass[1]
        num_finger = iclass[4:]

        source_class = os.path.join(source_dir, iclass)
        target_class = os.path.join(target_dir, iclass)
        if not os.path.exists(target_class):
            os.mkdir(target_class)
        source_train = os.path.join(source_class, 'train')
        source_test = os.path.join(source_class, 'test')
        target_train = os.path.join(target_class, '2015_train')
        target_test = os.path.join(target_class, '2015_test')

        """
        # Uncomment this if you have 6 images per class and hope to switch it to 4 train and 2 test per class
        
        import shutil
        if not os.path.exists(target_train):
            os.mkdir(target_train)
        if not os.path.exists(target_test):
            os.mkdir(target_test)
        for i in range(4):
            file_name = os.path.join(source_class, 'p{}.bmp'.format(i + 1))
            shutil.copy(file_name, target_train)
        print('writing to {}'.format(target_train))
        for i in range(2):
            file_name = os.path.join(source_class, 'p{}.bmp'.format(i + 5))
            shutil.copy(file_name, target_test)
        print('writing to {}'.format(target_test))
        """

        for i in range(6):
            # fingerprint preprocessing: IJCB2015->MHS, IJCB2017->HG, Frangi Filter
            # img = cv2.imread(os.path.join(source_class, '{}-{}-{}.jpg'.format(num_subject, num_finger, i+1)), flags=cv2.IMREAD_GRAYSCALE)  # physical
            img = cv2.imread(os.path.join(source_class, 'p{}.bmp'.format(i+1)), flags=cv2.IMREAD_GRAYSCALE)  # read input image
            img = cv2.resize(img, (224, 224))
            out = IJCB2015(img)
            cv2.imwrite(os.path.join(target_class, 'p{}.bmp'.format(i+1)), out)
            print('writing to: ' + os.path.join(target_class, 'p{}.bmp'.format(i+1)))

            # JPEG compression here
            # im = Image.open(os.path.join(source_class, 'p{}.bmp'.format(i+1)))
            # im.save(os.path.join(target_class, 'p{}.bmp'.format(i+1)), subsampling=0, format='JPEG', quality=compress)


if __name__ == '__main__':

    adv_types = ['boundary']
    for a_type in adv_types:
        finger_path = r'./datasets/final/{}'.format(a_type)
        current_path = './datasets/final/MHS/{}'.format(a_type)
        preprocess(finger_path, current_path)
