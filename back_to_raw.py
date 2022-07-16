import numpy as np
from FingerprintDataset import FingerprintTrain, Fingerprint_Seg
import segmentation
import torchvision
import cv2
import os


def export(path, img, adv):
    p, f = os.path.split(path[0])
    if p.find('database') != -1:
        new_path = p.replace('database', 'perturb_after_{}_fb'.format(adv))
    else:
        new_path = p.replace('evaluation', 'after_{}'.format(adv))
    filename = os.path.join(new_path, f)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, img)
    print('writing to' + filename)


position_database = np.load('./datasets/maps/position_database.npy', allow_pickle=True)
position_evaluation = np.load('./datasets/maps/position_evaluation.npy', allow_pickle=True)

adv = ['fingersafe']
for a in adv:
    raw_set = Fingerprint_Seg('./datasets/raw_pics/database')  # large
    adv_set = Fingerprint_Seg('./datasets/physical_square/perturb_{}'.format(a))  # 224*224
    clean_set = Fingerprint_Seg('./datasets/physical_square/database')  # square w*h
    raw_pics = raw_set.x_data
    adv_pics = adv_set.x_data
    clean_pics = clean_set.x_data
    unloader = torchvision.transforms.ToPILImage()
    trun_db_list = [
        2, 2, 2, 2, 2,
        2, 1.7, 1.7, 1.7, 1.7,
        1.7, 1.7, 1.7, 1, 1,
        1, 1, 1, 1, 1,
        2, 3, 2, 2, 2,
        2, 2, 2.5, 2, 2.2]
    trun_eval_list = [2, 2, 2, 2, 1.7,
                      1.7, 1.7, 1, 1, 1,
                      2, 1, 3, 2, 2,
                      2.1, 2, 2.5, 2.2, 1]
    position_list = []
    for idx in range(len(raw_pics)):
        img = adv_pics[idx]
        # return to raw pics
        img, noise = segmentation.reverse_segment(adv_pics[idx], raw_pics[idx], clean_pics[idx], position_database[idx])
        print('out')
        cv2.imwrite('./results/back.jpg', img)
        cv2.imwrite('./results/noise.jpg', noise)
        exit(0)

        if 70 <= idx < 100:  # evaluation s3
            img_seg = img
        else:
            img_seg, _, _, _ = segmentation.segment(img, trun=trun_eval_list[idx // 10])
        export(clean_set.paths[idx], img_seg, a)

