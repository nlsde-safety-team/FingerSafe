import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
import segmentation
from data import DataSampler, DataSampler_adv


class FingerprintDataset(Dataset):
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.length = 0

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length

    def load_image(self, files):
        print('load raw data')
        TARGET_IMG_SIZE = 224  # todo when the backbone is ScatNet, change it to 50
        img_to_tensor = transforms.ToTensor()
        gray = Gray()
        images = []
        label = []
        for (i_image, l) in files:
            img = Image.open(i_image)  # channels = 3/1
            img_resized = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))  # 224 * 224 * 3/224*224
            img_tensor = img_to_tensor(img_resized)
            images.append(img_tensor)  # 3*224*224 tensor/1*224*224
            label.append(l)
        label = torch.from_numpy(np.array(label)).long()
        return images, label

    def load_raw(self, files):
        print('load raw data')
        TARGET_IMG_SIZE = 224
        img_to_tensor = transforms.ToTensor()
        gray = Gray()
        images = []
        label = []
        for (i_image, l) in files:
            img = cv2.imread(i_image)  # channels = 3/1
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = np.transpose(img, (2, 0, 1))
            # img = Image.open(i_image)
            # img_tensor = img_to_tensor(img)
            # img_tensor = torch.from_numpy(img / 225.)
            images.append(img / 255.)  # 224*224*3 bgr array
            label.append(l)
        label = torch.from_numpy(np.array(label)).long()
        return images, label

    def load_and_segment(self, files):
        print('load raw data')
        TARGET_IMG_SIZE = 224
        img_to_tensor = transforms.ToTensor()
        unloader = transforms.ToPILImage()
        gray = Gray()
        images = []
        label = []
        position = []
        raw_pics = []
        masks = []
        for (i_image, l) in files:
            # img = Image.open(i_image)  # channels = 3/1
            # img_tensor = img_to_tensor(img)  # img is original picture, 3*w*h
            fin_img, fin_mask, pos, raw_pic = segmentation.segment(i_image)  # fin_img is RGB array, w*h*3
            fin_img = torch.from_numpy(np.transpose(fin_img, (2, 0, 1)))  # 3*w*h
            fin_mask = torch.ones(raw_pic[:, :, 0].shape)
            img_resized = unloader(fin_img)# .resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))  # 224 * 224 * 3/224*224
            masks_resized = unloader(fin_mask)# .resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            img_tensor = img_to_tensor(img_resized)
            masks_tensor = img_to_tensor(masks_resized)
            images.append(img_tensor)  # 3*224*224 tensor/1*224*224
            label.append(l)
            position.append(pos)
            raw_pics.append(raw_pic)
            masks.append(masks_tensor)
        label = torch.from_numpy(np.array(label)).long()
        return images, label, position, raw_pics, masks

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir, phase):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            # todo: if the orgniaztion of the files is like: 4 train / 2 test per class
            for root, _, fnames in sorted(os.walk(d)):
                 for fname in sorted(fnames):
                     if (fname.endswith('.jpg') or fname.endswith('.bmp')) and (root.count(phase) == 1):
                         path = os.path.join(root, fname)
                         item = (path, class_to_idx[target])
                         meshes.append(item)

            # todo: if the orgniaztion of the files is like: 6 images per class
            # for fname in sorted(os.listdir(d)):
            #     if fname.endswith('.jpg') or fname.endswith('.bmp'):
            #         path = os.path.join(d, fname)
            #         item = (path, class_to_idx[target])
            #         meshes.append(item)
        return meshes


class FingerprintTrain(FingerprintDataset):
    def __init__(self, root, phase='train'):
        FingerprintDataset.__init__(self)
        self.x_data, self.y_data = [], []
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(self.root, phase)
        self.paths = self.make_dataset_by_class(self.root, self.class_to_idx, phase)
        self.x_data, self.y_data = self.load_image(self.paths)
        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.paths[index]

    def __len__(self):
        return self.length


class FingerprintTest(FingerprintDataset):
    def __init__(self, root, phase='test'):
        FingerprintDataset.__init__(self)
        self.x_data, self.y_data = [], []
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(self.root, phase)
        self.paths = self.make_dataset_by_class(self.root, self.class_to_idx, phase)
        self.x_data, self.y_data = self.load_image(self.paths)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.paths[index]

    def __len__(self):
        self.length = len(self.y_data)
        return self.length


class FingerprintAdv(FingerprintDataset):
    def __init__(self, root, phase='train') -> object:
        FingerprintDataset.__init__(self)
        self.x_data, self.y_data = [], []
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(self.root, phase)
        self.paths = self.make_dataset_by_class(self.root, self.class_to_idx, phase)
        self.x_data, self.y_data = self.load_image(self.paths)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.paths[index]

    def __len__(self):
        self.length = len(self.y_data)
        return self.length


class Fingerprint_Seg(FingerprintDataset):
    def __init__(self, root, phase='train') -> object:
        FingerprintDataset.__init__(self)
        self.x_data, self.y_data, self.position, self.raw_pics = [], [], [], []
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(self.root, phase)
        self.paths = self.make_dataset_by_class(self.root, self.class_to_idx, phase)
        # self.x_data, self.y_data, self.position, self.raw_pics, self.masks = self.load_and_segment(self.paths)
        self.x_data, self.y_data = self.load_raw(self.paths)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]#, self.paths[index], \
               #self.position[index], self.raw_pics[index], self.masks[index]

    def __len__(self):
        self.length = len(self.y_data)
        return self.length


class Fingerprint_Mask(FingerprintDataset):
    """
    load seg_square and mask, resize
    """
    def __init__(self, root, root_mask=None, phase='train') -> object:
        FingerprintDataset.__init__(self)
        self.x_data, self.y_data = [], []
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(self.root, phase)
        self.paths = self.make_dataset_by_class(self.root, self.class_to_idx, phase)
        self.paths_mask = self.make_dataset_by_class(root_mask, self.class_to_idx, phase)
        self.x_data, self.y_data = self.load_image(self.paths)
        self.masks, _ = self.load_image(self.paths_mask)
        # self.x_data, self.y_data = self.load_raw(self.paths)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.paths[index], self.masks[index]

    def __len__(self):
        self.length = len(self.y_data)
        return self.length


class FingerprintMix(FingerprintDataset):
    def __init__(self, root, clean='train', dirty='perturb_fingeradv', ratio=0.5):
        import random
        import math
        FingerprintDataset.__init__(self)
        self.x_data, self.y_data = [], []
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(self.root)
        self.paths_clean = self.make_dataset_by_class(self.root, self.class_to_idx, clean)
        self.paths_dirty = self.make_dataset_by_class(self.root, self.class_to_idx, dirty)
        self.paths = self.paths_clean
        line = [x for x in range(0, len(self.paths))]
        random.shuffle(line)
        line = line[0: math.ceil(len(self.paths) * ratio)]
        for i in range(len(line)):
            if i in line:
                print(i)
                self.paths[i] = self.paths_dirty[i]
        self.x_data, self.y_data = self.load_image(self.paths)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        self.length = len(self.y_data)
        return self.length

import pickle
from PIL import Image
class FingerprintName(FingerprintDataset):
    def __init__(self, root, phase='train', name=None):  # note: finish dataloader
        FingerprintDataset.__init__(self)
        if phase == 'train':
            self.class_num = 268
        else:
            self.class_num = 68
        self.image_per_class = 6
        read_mode = 'rgb'
        self.length = self.class_num * self.image_per_class * self.image_per_class
        self.root = root
        # self.fingerprint_raw, self.subject_id, self.fingerprint_id = self.get_dataset(self.root, phase, read_mode, name=name)
        self.get_dataset(self.root, phase, read_mode, name=name)
        # print(self.fingerprint_raw.shape)
        # self.fingerprint_raw = self.fingerprint_raw.transpose((0, 3, 1, 2))

        self.path = self.get_path(self.root, phase)

    def __getitem__(self, index):
        index_fig = index * 6
        return self.fingerprint_raw[index_fig:index_fig + 6, ...].astype('float32'), self.path[index_fig:index_fig + 6]

    def __len__(self):
        return int(self.fingerprint_raw.shape[0] / 6)

    def get_path(self, path, method):
        TARGET_IMG_SIZE = 224 
        dir = os.path.join(path, method)
        path = []
        for fpath, dirname, fnames in os.walk(dir):
            for fid in fnames:
                if fid.endswith('bmp'):
                    path.append(os.path.join(fpath, fid))
        return path

    # def get_dataset(self, path, method, read_mode, name):  # physical
    #     img_to_tensor = transforms.ToTensor()
    #     TARGET_IMG_SIZE = 224
    #     dir = os.path.join(path)  # todo sample directory
    #     data = []
    #     subject_id = []
    #     fingerprint_id = []
    #     classes=[]
    #     datapath_all = {}
    #     class_end = 0
    #     class_id = 0
    #     for fpath, dirname, fnames in os.walk(dir):
    #         i=0
    #         for fid in fnames:
    #             i += 1
    #             if fid.endswith('bmp') or fid.endswith('jpg'):
    #                 if method == 'test':
    #                     fid_list = fid.split('_')
    #                 elif method == 'train':
    #                     # fid_list = fid.split('.')
    #                     # path_list = os.path.split(fpath)
    #                     fid_list = fid.split('_')
    #                 else:
    #                     fid_list = fid.split('-')
    #                 if method == 'test':
    #                     subject = int(fid_list[0]) - 268
    #                     finger_id = list(fid_list[1])[0]
    #                 elif method == 'train':
    #                     # subject = path_list[-1][1:]
    #                     # finger_id = int(fid_list[0][1:])
    #                     subject = int(fid_list[0])
    #                     finger_id = list(fid_list[1])[0]
    #                 else:
    #                     finger_id = fid_list[0] + '-' + fid_list[1]
    #                     pic_id = fid_list[2].split('.')[0]
    #                 if not finger_id in fingerprint_id:
    #                     class_end += 1
    #                     class_id = class_end
    #                     classes.append(finger_id)
    #                 else:
    #                     class_id = classes.index(finger_id) + 1
    #                     # print(class_id)
    #
    #
    #                 fingerprint_id.append(finger_id)
    #                 if class_id in datapath_all:
    #                     subject_dict = datapath_all.get(class_id)
    #                     # print(subject_dict)
    #                 else:
    #                     subject_dict = {}
    #
    #                 subject_dict[int(pic_id)] = os.path.join(fpath, fid)
    #                 # print(subject_dict)
    #
    #                 datapath_all[class_id] = subject_dict
    #             # if i==3:
    #             #     break
    #
    #     # save
    #
    #     print(fingerprint_id)
    #     print(classes)
    #     f = open('./datapaths/datapath_{}_'.format(name) + 'train' + '.pkl', 'wb')
    #     pickle.dump(datapath_all, f)
    #     print(datapath_all)
    #     print('./datapaths/datapath_{}_'.format(name) + 'train' + '.pkl')
    #     # exit(0)
    #
    #     data = np.asarray(data)  # (1608, 224, 352, 3)
    #     #print(datapath_all)
    #     '''
    #     data_new = torch.zeros(data.shape[0], data.shape[1], data.shape[2])
    #     for i in range(data.shape[0]):
    #         data_new[i, ...] = gray(img_to_tensor(data[i, ...])).squeeze()
    #     data = data_new  # (1608, 224, 352, 1)
    #     '''
    #
    #     subject_id = np.asarray(subject_id)  # (1608,)
    #     fingerprint_id = np.asarray(fingerprint_id)  # (1608,)
    #     return data, subject_id, fingerprint_id

    def get_dataset(self, path, method, read_mode, name=None):  # polyU
        img_to_tensor = transforms.ToTensor()
        TARGET_IMG_SIZE = 224
        dir = os.path.join(path)  # todo sample directory
        data = []
        subject_id = []
        fingerprint_id = []
        datapath_all = {}
        for fpath, dirname, fnames in os.walk(dir):
            for fid in fnames:
                if fid.endswith('bmp') or fid.endswith('jpg'):
                    if method == 'test':
                        fid_list = fid.split('_')
                    elif method == 'train':
                        # fid_list = fid.split('.')
                        # path_list = os.path.split(fpath)
                        fid_list = fid.split('_')
                    else:
                        fid_list = fid.split('-')

                    # if read_mode == 'rgb':
                    #     image = cv2.imread(os.path.join(fpath, fid))
                    # else:
                    #     image = cv2.imread(os.path.join(fpath, fid), cv2.IMREAD_GRAYSCALE)
                    # image = cv2.resize(image, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
                    # image = image / 255
                    image = Image.open(os.path.join(fpath, fid))
                    image = image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
                    image = img_to_tensor(image)
                    data.append(np.asarray(image))
                    if method == 'test':
                        subject = int(fid_list[0]) - 268
                        finger_id = list(fid_list[1])[0]
                    else:
                        # subject = path_list[-1][1:]
                        # finger_id = int(fid_list[0][1:])
                        subject = int(fid_list[0])
                        finger_id = list(fid_list[1])[0]
                    subject_id.append(subject)

                    fingerprint_id.append(int(finger_id))
                    if subject in datapath_all:
                        subject_dict = datapath_all.get(subject)
                    else:
                        subject_dict = {}

                    subject_dict[int(finger_id)] = os.path.join(fpath, fid)
                    datapath_all[subject] = subject_dict

        # save

        f = open('./datapaths/datapath_{}_'.format(name) + 'test' + '.pkl', 'wb')
        pickle.dump(datapath_all, f)
        print(datapath_all)
        print('./datapaths/datapath_{}_'.format(name) + 'test' + '.pkl')
        #
        # data = np.asarray(data)  # (1608, 224, 352, 3)
        # print(datapath_all)
        # '''
        # data_new = torch.zeros(data.shape[0], data.shape[1], data.shape[2])
        # for i in range(data.shape[0]):
        #     data_new[i, ...] = gray(img_to_tensor(data[i, ...])).squeeze()
        # data = data_new  # (1608, 224, 352, 1)
        # '''
        #
        # subject_id = np.asarray(subject_id)  # (1608,)
        # fingerprint_id = np.asarray(fingerprint_id)  # (1608,)
        # return data, subject_id, fingerprint_id

class FingerprintVeri(FingerprintDataset):
    def __init__(self, pkl_path, num):
        FingerprintDataset.__init__(self)
        self.class_num = 268
        self.image_per_class = 6
        # self.length = self.class_num * self.image_per_class * self.image_per_class * 2
        self.root = pkl_path
        clean_pkl = './datapaths/datapath_MHS_clean_test.pkl'
        data_sampler = DataSampler_adv(num, 4, clean_pkl, self.root, mode='rgb')
        np.random.seed(0)
        siamese_1, siamese_2, label, ad1, ad2 = data_sampler.sample(flag=False)
        valid_1 = siamese_1
        valid_2 = siamese_2
        valid_issame = label
        for i_pair in range(40):
            siamese_1, siamese_2, label, ad1, ad2 = data_sampler.sample(flag=False)
            valid_1 = torch.cat([valid_1, siamese_1], dim=0)  # 201 * batch_size
            valid_2 = torch.cat([valid_2, siamese_2], dim=0)  # 201 * batch_size
            valid_issame = torch.cat([valid_issame, label], dim=0)  # pairs
        self.pair_1 = valid_1
        self.pair_2 = valid_2
        self.pair_issame = valid_issame

    def __getitem__(self, index):
        # output: image pair, label of 0 or 1 (0 as positive and 1 as negative)
        return self.pair_1[index], self.pair_2[index], self.pair_issame[index]

    def __len__(self):
        return len(self.pair_issame)



class Gray(object):

    def __call__(self, tensor):  # tensor: 3 * w * h
        # TODO: make efficient
        if tensor.shape[0] == 1:
            return tensor
        _, w, h = tensor.shape
        R = tensor[0]
        G = tensor[1]
        B = tensor[2]
        tensor[0] = 0.299 * R + 0.587 * G + 0.114 * B
        tensor = tensor[0]
        tensor = tensor.view(1, w, h)
        return tensor


if __name__ == '__main__':
    FingerprintTrain('datasets/fingerprint_colored')
