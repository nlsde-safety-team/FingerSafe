import os
import argparse
import cv2
from models import Fingerprint_enhancer_our
import fingerprint_enhancer
import torchvision.transforms as transforms
from frangi_filter import FrangiFilter2D
from PIL import Image
import shutil
from FingerSafe import Gray

os.environ["CUDA_VISIBLE_DEVICES"]="2"


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


def preprocess(source_dir, target_dir, convert, pre):
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


        # preprocessors
        if 1 <= convert <= 3:
            for i in range(6):
                # img = cv2.imread(os.path.join(source_class, '{}-{}-{}.jpg'.format(num_subject, num_finger, i+1)), flags=cv2.IMREAD_GRAYSCALE)  # physical
                img = cv2.imread(os.path.join(source_class, 'p{}.bmp'.format(i+1)), flags=cv2.IMREAD_GRAYSCALE)  # read input image
                img = cv2.resize(img, (224, 224))
                if convert == 1:
                    out = IJCB2015(img)
                elif convert == 2:
                    out = IJCB2017(img)
                elif convert == 3:
                    out = Frangi_Filter(img)
                cv2.imwrite(os.path.join(target_class, 'p{}.bmp'.format(i+1)), out)
                print('writing to: ' + os.path.join(target_class, 'p{}.bmp'.format(i+1)))

        # todo comperssion
        elif convert == 4:
            for i in range(6):
                im = Image.open(os.path.join(source_class, 'p{}.bmp'.format(i+1)))
                im.save(os.path.join(target_class, 'p{}.bmp'.format(i+1)), subsampling=0, format='JPEG', quality=30)

        # todo split the dataset：6 images per class --> 4 train / 2 test per class
        elif 5 <= convert <= 7:
            if convert == 5:  # for identification (RGB)
                target_train = os.path.join(target_class, 'training')
                target_test = os.path.join(target_class, 'testing')
            elif convert == 6:  # for identification (preprocessed)
                target_train = os.path.join(target_class, '{}_train'.format(pre))
                target_test = os.path.join(target_class, '{}_test'.format(pre))
            elif convert == 7:  # for training stage
                target_train = os.path.join(target_class, 'train')
                target_test = os.path.join(target_class, 'test')
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
        else:
            print("convert must be between 1 and 7!")
            exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path', '-op', type=str,
                        help='original path of data', default="./datasets/final/veri_sample_fingersafe_gamma_10")
    parser.add_argument('--new_path', '-np', type=str,
                        help='new path of data',
                        default="./datasets/final/HG/sample_fingersafe_gamma_10")
    parser.add_argument('--convert', '-c', type=int,
                        help='choose how to convert, must be between 1 and 7',
                        default=2)
    parser.add_argument('--pre', type=str,
                        help='when convert=6, choose which preprocessor is used, must be "2015", "2017" or "frangi" ',
                        default="")
    args = parser.parse_args()

    original_path = args.original_path
    new_path = args.new_path
    convert = args.convert
    pre = args.pre
    preprocess(original_path, new_path, convert, pre)