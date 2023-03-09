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

# position_s4s5 = np.load('./datasets/maps/s4s5.npy', allow_pickle=True)
#
# position_base_s4 = np.vstack((position_s4s5[:10], position_s4s5[20:60]))
# position_eval_s4 = np.vstack((position_s4s5[10:20], position_s4s5[60:100]))
# position_base_s5 = np.vstack((position_s4s5[100:110], position_s4s5[120:160]))
# position_eval_s5 = np.vstack((position_s4s5[110:120], position_s4s5[160:200]))
# position_database = np.vstack((position_database, position_base_s4))
# position_database = np.vstack((position_database, position_base_s5))
# position_evaluation = np.vstack((position_evaluation, position_eval_s4))
# position_evaluation = np.vstack((position_evaluation, position_eval_s5))
#
# np.save('./datasets/maps/position_database.npy', position_database)
# np.save('./datasets/maps/position_evaluation.npy', position_evaluation)
# exit(0)
adv = ['fingersafe']
for a in adv:
    raw_set = Fingerprint_Seg('./datasets/raw_pics/database')  # large
    adv_set = Fingerprint_Seg('./datasets/physical_square/perturb_{}'.format(a))  # 224*224
    clean_set = Fingerprint_Seg('./datasets/physical_square/database')  # square w*h
    raw_pics = raw_set.x_data
    adv_pics = adv_set.x_data
    clean_pics = clean_set.x_data
    unloader = torchvision.transforms.ToPILImage()
    # trun_list = [2, 2, 3, 2, 2,
    #              2, 1, 3, 2, 2,
    #              2, 2.1, 2, 2.5, 2,
    #              2.2, 2, 2.5, 2.2, 1]
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

        # segment again
        # if 130 <= idx < 200:  # database s3
        if 70 <= idx < 100:  # evaluation s3
            img_seg = img
        else:
            img_seg, _, _, _ = segmentation.segment(img, trun=trun_eval_list[idx // 10])

        # if idx < 60:  # s1, trun_index=2
        #     img_seg, _, _, _ = segmentation.segment(img, trun=2)
        #     print('s1')
        # elif idx < 130:  # s2,trun_index=1.7
        #     img_seg, _, _, _ = segmentation.segment(img, trun=1.7)
        #     print('s2')
        # else:  # s3
        #     img_seg = img
        #     print('s3')

        export(clean_set.paths[idx], img_seg, a)
    #     img_seg, mask, pos, _ = segmentation.segment(raw_pics[idx], trun=trun_list[idx // 10])
    #     print('f'+ str(idx // 10))
    #     print(trun_list[idx // 10])
    #     position_list.append(pos)
    #     export(raw_set.paths[idx], img_seg, name='_seg')
    #     export(raw_set.paths[idx], mask, name='_mask')
    # np.save('datasets/maps/s4s5.npy', position_list)

