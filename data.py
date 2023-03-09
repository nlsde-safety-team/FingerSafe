import torch.utils.data.dataset as dataset
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import os

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# training_sample_count = 4000
# validation_sample_count = 1000
# test_sample_count = 1000

class DataSampler:
    def __init__(self, train_num, batch_size, pkl_path, mode, unlearnable=False):
        self.train_num = train_num
        self.load_dataset(pkl_path)
        self.batch_size = batch_size
        self.mode = mode
        self.unlearnable = unlearnable

    def load_dataset(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.files_lst = pickle.load(f)

    def read_image(self, name, mode):
        from PIL import Image
        if mode == 'rgb':
            # img = cv2.imread(name)
            img = Image.open(name)
        else:
            img = Image.open(name)
            # img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        return img

    def sample(self, flag=False):
        """
        batch_size: positive and negative
        """
        # ############################################################################
        # sample postive samples
        # ############################################################################
        # label = 0, when sampling from the same class
        id_p1 = []
        id_f1 = []
        id_p2 = []
        id_f2 = []
        ad1, ad2 = [], []

        TARGET_IMG_SIZE = 224
        pos_label = torch.zeros([self.batch_size, 1], dtype=torch.float32)
        # train_num: 4000, batch_size:4, train_num=268

        # np.random.seed(0)
        pos_index = np.random.randint(0, self.train_num, [self.batch_size, 1]) + 1
        # pos_index = np.array([[1], [1], [1], [1]])
        # len(self.files_lst[pos_index[0, 0]]) indicates how many index are in finger
        if flag:  # flag=true, when sampling from training set; flag=False, when sampling from testing set
            each_index = np.random.randint(0, len(self.files_lst[pos_index[0, 0]]), [1, 2]) + 5
        else:
            each_index = np.random.randint(0, len(self.files_lst[pos_index[0, 0]]), [1, 2]) + 1

        id_p1.append(pos_index[0, 0])
        id_f1.append(each_index[0, 0])
        id_p2.append(pos_index[0, 0])
        id_f2.append(each_index[0, 1])

        address_1 = self.files_lst[pos_index[0, 0]][each_index[0, 0]]
        ad1.append(address_1)

        siamese_1_img_batch = self.read_image(address_1, self.mode)
        siamese_1_img_batch = siamese_1_img_batch.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_1_img_batch, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        siamese_1_img_batch = torch.unsqueeze(transforms.ToTensor()(siamese_1_img_batch), dim=0)

        address_2 = self.files_lst[pos_index[0, 0]][each_index[0, 1]]
        ad2.append(address_2)
        siamese_2_img_batch = self.read_image(address_2, self.mode)
        siamese_2_img_batch = siamese_2_img_batch.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))#cv2.resize(siamese_2_img_batch, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        siamese_2_img_batch = torch.unsqueeze(transforms.ToTensor()(siamese_2_img_batch), dim=0)

        # np.random.seed(0)
        for i in range(1, self.batch_size):
            if flag:
                each_index = np.random.randint(0, len(self.files_lst[pos_index[i, 0]]), [1, 2]) + 5
            else:
                each_index = np.random.randint(0, len(self.files_lst[pos_index[i, 0]]), [1, 2]) + 1

            id_p1.append(pos_index[i, 0])
            id_f1.append(each_index[0, 0])
            id_p2.append(pos_index[i, 0])
            id_f2.append(each_index[0, 1])

            address_1 = self.files_lst[pos_index[i, 0]][each_index[0, 0]]
            ad1.append(address_1)
            siamese_1_pos = self.read_image(address_1, self.mode)
            siamese_1_pos = siamese_1_pos.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_1_pos, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            siamese_1_pos = torch.unsqueeze(transforms.ToTensor()(siamese_1_pos), dim=0)
            # ToTensor: normalized to [0, 1] division 255
            address_2 = self.files_lst[pos_index[i, 0]][each_index[0, 1]]
            ad2.append(address_2)
            siamese_2_pos = self.read_image(address_2, self.mode)
            siamese_2_pos = siamese_2_pos.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_2_pos, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            siamese_2_pos = torch.unsqueeze(transforms.ToTensor()(siamese_2_pos), dim=0)
            # print("address_1 = ", address_1)
            # print("address_2 = ", address_2)

            # print(siamese_1_pos.shape)
            # print(siamese_2_pos.shape)

            siamese_1_img_batch = torch.cat([siamese_1_img_batch, siamese_1_pos], dim=0)
            siamese_2_img_batch = torch.cat([siamese_2_img_batch, siamese_2_pos], dim=0)

        # ############################################################################
        # sample negative samples
        # ############################################################################
        # label = 1 when sampling from different classes
        neg_label = torch.ones([self.batch_size, 1], dtype=torch.float32)
        label_tensor = torch.cat([pos_label, neg_label], dim=0)
        # np.random.seed(0)
        for i in range(self.batch_size):
            while True:
                neg_index = np.random.randint(0, self.train_num, [1, 2]) + 1
                if neg_index[0, 0] != neg_index[0, 1]:
                    break
            if flag:
                index_1 = np.random.randint(0, len(self.files_lst[neg_index[0, 0]]), [1]) + 5
            else:
                index_1 = np.random.randint(0, len(self.files_lst[neg_index[0, 0]]), [1]) + 1

            id_p1.append(neg_index[0, 0])
            id_f1.append(index_1[0])

            address_1 = self.files_lst[neg_index[0, 0]][index_1[0]]
            ad1.append(address_1)
            siamese_1_neg = self.read_image(address_1, self.mode)
            siamese_1_neg = siamese_1_neg.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_1_neg, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            siamese_1_neg = torch.unsqueeze(transforms.ToTensor()(siamese_1_neg), dim=0)

            if flag:
                index_2 = np.random.randint(0, len(self.files_lst[neg_index[0, 1]]), [1]) + 5
            else:
                index_2 = np.random.randint(0, len(self.files_lst[neg_index[0, 1]]), [1]) + 1

            id_p2.append(neg_index[0, 1])
            id_f2.append(index_2[0])

            address_2 = self.files_lst[neg_index[0, 1]][index_2[0]]
            ad2.append(address_2)
            siamese_2_neg = self.read_image(address_2, self.mode)
            siamese_2_neg = siamese_2_neg.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_2_neg, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            siamese_2_neg = torch.unsqueeze(transforms.ToTensor()(siamese_2_neg), dim=0)
            # print("address_1 = ", address_1)
            # print("address_2 = ", address_2)

            # print(siamese_1_neg.shape)
            # print(siamese_2_neg.shape)

            siamese_1_img_batch = torch.cat([siamese_1_img_batch, siamese_1_neg], dim=0)
            siamese_2_img_batch = torch.cat([siamese_2_img_batch, siamese_2_neg], dim=0)
        if self.unlearnable:
            return siamese_1_img_batch, siamese_2_img_batch, np.asarray(id_p1) - 1, np.asarray(id_f1) - 1, np.asarray(id_p2) - 1, np.asarray(id_f2) - 1, label_tensor
        else:
            return siamese_1_img_batch, siamese_2_img_batch, label_tensor, ad1, ad2

# DataSampler_adv: sampling pairs of fingerprint from clean & adv
class DataSampler_adv:
    def __init__(self, train_num, batch_size, pkl_path_clean, pkl_path_adv, mode, unlearnable=False):
        self.train_num = train_num
        self.load_dataset(pkl_path_clean, pkl_path_adv)
        self.batch_size = batch_size
        self.mode = mode
        self.unlearnable = unlearnable

    def load_dataset(self, pkl_path_clean, pkl_path_adv):
        # self.file_lst_clean = np.load(pkl_path_clean, allow_pickle=True)
        # self.file_lst_adv = np.load(pkl_path_adv, allow_pickle=True)
        with open(pkl_path_clean, "rb") as f:
            self.files_lst_clean = pickle.load(f)
        with open(pkl_path_adv, "rb") as f:
            self.files_lst_adv = pickle.load(f)

    def read_image(self, name, mode):
        from PIL import Image
        if mode == 'rgb':
            # img = cv2.imread(name)
            img = Image.open(name)
        else:
            img = Image.open(name)
            # img.convert('L')
            # print(img)
            # img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        return img

    def sample(self, flag=False):
        """
        batch_size: positive and negative
        """
        # ############################################################################
        # sample postive samples
        # ############################################################################
        id_p1 = []
        id_f1 = []
        id_p2 = []
        id_f2 = []
        ad2 = []
        ad1 = []

        TARGET_IMG_SIZE = 224
        pos_label = torch.zeros([self.batch_size, 1], dtype=torch.float32)

        # train_num: 4000, batch_size:32
        # np.random.seed(0)
        pos_index = np.random.randint(0, self.train_num, [self.batch_size, 1]) + 1
        # pos_index = np.array([[1], [1], [1], [1]])
        # len(self.files_lst[pos_index[0, 0]]) indicates how many index are in finger
        # np.random.seed(0)

        each_index = np.random.randint(0, len(self.files_lst_clean[pos_index[0, 0]]), [1, 2]) + 1
        # each_index = np.array([[1, 2]])

        id_p1.append(pos_index[0, 0])
        id_f1.append(each_index[0, 0])
        id_p2.append(pos_index[0, 0])
        id_f2.append(each_index[0, 1])

        address_1 = self.files_lst_clean[pos_index[0, 0]][each_index[0, 0]]
        ad1.append(address_1)

        siamese_1_img_batch = self.read_image(address_1, self.mode)
        siamese_1_img_batch = siamese_1_img_batch.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_1_img_batch, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        siamese_1_img_batch = torch.unsqueeze(transforms.ToTensor()(siamese_1_img_batch), dim=0)
        if siamese_1_img_batch.shape[1] == 3 and self.mode != 'rgb':
            siamese_1_img_batch = 0.299 * siamese_1_img_batch[:, 0, :, :] + 0.587 * siamese_1_img_batch[:, 0, :, :] + 0.114 * siamese_1_img_batch[:, 0, :, :]
            siamese_1_img_batch = torch.unsqueeze(siamese_1_img_batch, dim=1)

        address_2 = self.files_lst_adv[pos_index[0, 0]][each_index[0, 1]]
        ad2.append(address_2)
        siamese_2_img_batch = self.read_image(address_2, self.mode)
        siamese_2_img_batch = siamese_2_img_batch.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_2_img_batch, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        siamese_2_img_batch = torch.unsqueeze(transforms.ToTensor()(siamese_2_img_batch), dim=0)
        if siamese_2_img_batch.shape[1] == 3 and self.mode != 'rgb':
            siamese_2_img_batch = 0.299 * siamese_2_img_batch[:, 0, :, :] + 0.587 * siamese_2_img_batch[:, 0, :, :] + 0.114 * siamese_2_img_batch[:, 0, :, :]
            siamese_2_img_batch = torch.unsqueeze(siamese_2_img_batch, dim=1)

        # np.random.seed(0)
        for i in range(1, self.batch_size):
            each_index = np.random.randint(0, len(self.files_lst_clean[pos_index[i, 0]]), [1, 2]) + 1
            # each_index = np.array([[3, 4]])

            id_p1.append(pos_index[i, 0])
            id_f1.append(each_index[0, 0])
            id_p2.append(pos_index[i, 0])
            id_f2.append(each_index[0, 1])

            address_1 = self.files_lst_clean[pos_index[i, 0]][each_index[0, 0]]
            ad1.append(address_1)
            siamese_1_pos = self.read_image(address_1, self.mode)
            siamese_1_pos = siamese_1_pos.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_1_pos, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            siamese_1_pos = torch.unsqueeze(transforms.ToTensor()(siamese_1_pos), dim=0)

            if siamese_1_pos.shape[1] == 3 and self.mode != 'rgb':
                siamese_1_pos = 0.299 * siamese_1_pos[:, 0, :, :] + 0.587 * siamese_1_pos[:, 0, :, :] + 0.114 * siamese_1_pos[:, 0, :, :]
                siamese_1_pos = torch.unsqueeze(siamese_1_pos, dim=1)

            # ToTensor: normalized to [0, 1] division 255
            address_2 = self.files_lst_adv[pos_index[i, 0]][each_index[0, 1]]
            ad2.append(address_2)
            siamese_2_pos = self.read_image(address_2, self.mode)
            siamese_2_pos = siamese_2_pos.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# siamese_2_pos.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_2_pos, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            siamese_2_pos = torch.unsqueeze(transforms.ToTensor()(siamese_2_pos), dim=0)

            if siamese_2_pos.shape[1] == 3 and self.mode != 'rgb':
                siamese_2_pos = 0.299 * siamese_2_pos[:, 0, :, :] + 0.587 * siamese_2_pos[:, 0, :, :] + 0.114 * siamese_2_pos[:, 0, :, :]
                siamese_2_pos = torch.unsqueeze(siamese_2_pos, dim=1)

            # print("address_1 = ", address_1)
            # print("address_2 = ", address_2)

            # print(siamese_1_pos.shape)
            # print(siamese_2_pos.shape)

            siamese_1_img_batch = torch.cat([siamese_1_img_batch, siamese_1_pos], dim=0)
            siamese_2_img_batch = torch.cat([siamese_2_img_batch, siamese_2_pos], dim=0)

        # ############################################################################
        # sample negative samples
        # ############################################################################
        neg_label = torch.ones([self.batch_size, 1], dtype=torch.float32)
        label_tensor = torch.cat([pos_label, neg_label], dim=0)
        # np.random.seed(0)
        for i in range(self.batch_size):
            while True:
                neg_index = np.random.randint(0, self.train_num, [1, 2]) + 1
                if neg_index[0, 0] != neg_index[0, 1]:
                    break
            index_1 = np.random.randint(0, len(self.files_lst_clean[neg_index[0, 0]]), [1]) + 1

            id_p1.append(neg_index[0, 0])
            id_f1.append(index_1[0])

            address_1 = self.files_lst_clean[neg_index[0, 0]][index_1[0]]
            ad1.append(address_1)
            siamese_1_neg = self.read_image(address_1, self.mode)
            siamese_1_neg = siamese_1_neg.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_1_neg, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            siamese_1_neg = torch.unsqueeze(transforms.ToTensor()(siamese_1_neg), dim=0)

            if siamese_1_neg.shape[1] == 3 and self.mode != 'rgb':
                siamese_1_neg = 0.299 * siamese_1_neg[:, 0, :, :] + 0.587 * siamese_1_neg[:, 0, :, :] + 0.114 * siamese_1_neg[:, 0, :, :]
                siamese_1_neg = torch.unsqueeze(siamese_1_neg, dim=1)

            index_2 = np.random.randint(0, len(self.files_lst_adv[neg_index[0, 1]]), [1]) + 1

            id_p2.append(neg_index[0, 1])
            id_f2.append(index_2[0])

            address_2 = self.files_lst_adv[neg_index[0, 1]][index_2[0]]
            ad2.append(address_2)
            siamese_2_neg = self.read_image(address_2, self.mode)
            siamese_2_neg = siamese_2_neg.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))# cv2.resize(siamese_2_neg, dsize=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            siamese_2_neg = torch.unsqueeze(transforms.ToTensor()(siamese_2_neg), dim=0)
            
            if siamese_2_neg.shape[1] == 3 and self.mode != 'rgb':
                siamese_2_neg = 0.299 * siamese_2_neg[:, 0, :, :] + 0.587 * siamese_2_neg[:, 0, :, :] + 0.114 * siamese_2_neg[:, 0, :, :]
                siamese_2_neg = torch.unsqueeze(siamese_2_neg, dim=1)

            # print("address_1 = ", address_1)
            # print("address_2 = ", address_2)

            # print(siamese_1_neg.shape)
            # print(siamese_2_neg.shape)

            siamese_1_img_batch = torch.cat([siamese_1_img_batch, siamese_1_neg], dim=0)
            siamese_2_img_batch = torch.cat([siamese_2_img_batch, siamese_2_neg], dim=0)
        if self.unlearnable:
            return siamese_1_img_batch, siamese_2_img_batch, np.asarray(id_p1) - 1, np.asarray(id_f1) - 1, np.asarray(id_p2) - 1, np.asarray(id_f2) - 1, label_tensor
        else:
            return siamese_1_img_batch, siamese_2_img_batch, label_tensor, ad1, ad2
