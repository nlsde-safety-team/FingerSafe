import numpy as np
import os
import argparse
import cv2

'''
Change the dataset from classification to verification.

Task: 
in classification we split the fingerprint collected from each subject to train/veri_test.
eg, subject 1 have 6 fingerprints, split it into 4 for training and 2 for veri_test

in verification we split the fingerprint by subject id.
eg, 336 subjects, so we split 269 subject as train and the rest for veri_test. the specific number might vary
due to train_test ratio.

Finger Path: place we put the dataset for classification
current dir: place we got our new data for verification

So the final datasets folder should contain fingerprint_classification and fingerprint_verification
'''


def cla2veri(finger_path, current_path, convert):
    if not os.path.exists(current_path):
        os.mkdir(current_path)

    if convert == 1:
        for dirname in os.listdir(finger_path):
            subject_id = dirname[1:]
            for fname in os.listdir(os.path.join(finger_path, dirname)):
                fig_id = fname[1]
                fig_name = subject_id + '_' + fig_id + '.bmp'
                # img = cv2.imread(os.path.join(finger_path, dirname, fname), cv2.IMREAD_GRAYSCALE)  # todo gray
                img = cv2.imread(os.path.join(finger_path, dirname, fname))  # todo RGB
                cv2.imwrite(os.path.join(current_path, fig_name), img)
                print('writing to' + os.path.join(current_path, fig_name))
    elif convert == 2:  # specially for training stage
        for dirname in os.listdir(finger_path):
            subject_id = dirname[1:]
            for fname in os.listdir(os.path.join(finger_path, dirname, 'test')):
                fig_id = fname[1]
                fig_name = subject_id + '_' + fig_id + '.bmp'
                # img = cv2.imread(os.path.join(finger_path, dirname, fname), cv2.IMREAD_GRAYSCALE)  # todo gray
                img = cv2.imread(os.path.join(finger_path, dirname, 'test', fname))  # todo RGB
                cv2.imwrite(os.path.join(current_path, fig_name), img)
                print('writing to' + os.path.join(current_path, fig_name))

    elif convert == 3:
        for dirname in os.listdir(finger_path):
            print(dirname)
            for fname in os.listdir(os.path.join(finger_path, dirname)):
                img = cv2.imread(os.path.join(finger_path, dirname, fname))
                cv2.imwrite(os.path.join(current_path, fname), img)
                print('writing to' + os.path.join(current_path, fname))

    elif convert == 4:
        for filename in os.listdir(finger_path):
            subject_id = filename[0:3]
            fig_id = filename[4]
            subject_dir = os.path.join(current_path, 'f' + subject_id)
            if not os.path.exists(subject_dir):
                os.mkdir(subject_dir)
            fig_name = 'p' + fig_id + '.bmp'
            img = cv2.imread(os.path.join(finger_path, filename), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(subject_dir, fig_name), img)
            print('writing to ' + os.path.join(subject_dir, fig_name))

    elif convert == 5:
        for filename in os.listdir(finger_path):
            s = filename.split('.')
            n = s[0].split('-')
            subject_id = n[0]
            fig_id = n[1]
            pic_id = n[2]
            subject_dir = os.path.join(current_path, 'f' + subject_id + '-' + 's' + fig_id)
            if not os.path.exists(subject_dir):
                os.mkdir(subject_dir)
            fig_name = subject_id + '-' + fig_id + '-' + pic_id + '.jpg'
            img = cv2.imread(os.path.join(finger_path, filename))
            cv2.imwrite(os.path.join(subject_dir, fig_name), img)
            print('writing to ' + os.path.join(subject_dir, fig_name))
    else:
        print("convert must be between 1 and 5!")
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path', '-op', type=str,
                        help='original path of data', default="./datasets/final/MHS/sample_fingersafe_gamma_10")
    parser.add_argument('--new_path', '-np', type=str,
                        help='new path of data',
                        default="./datasets/final/fingersafe_gamma_10_test")
    parser.add_argument('--convert', '-c', type=int,
                        help='choose how to convert, must be 1~5',
                        default=1)
    args = parser.parse_args()

    original_path = args.original_path
    new_path = args.new_path
    convert = args.convert
    cla2veri(original_path, new_path, convert)
