import shutil

import torch, torchvision
import time
import numpy as np
from torch.utils.data import DataLoader
from FingerprintDataset import FingerprintAdv, Fingerprint_Mask, FingerprintName
from criterion import Criterion
import os
from torchvision import transforms as trans
from models.inception_resnet_v1 import ResNet as Model
from models.inception_resnet_v1 import Inceptionv3, DenseNet
from FingerSafe import FingerSafe
import random
import segmentation
from L_orientation import ridge_orient


def export(paths, images, position=None, raw_pics=None):
    unloader = torchvision.transforms.ToPILImage()
    # rgb2bgr = [2, 1, 0]
    # images = images[:, rgb2bgr, :, :]
    for idx in range(len(images)):
        p, f = os.path.split(paths[0][idx])
        if p.find('train') != -1:
            new_path = p.replace('train', 'perturb_tmp_I')
        else:
            new_path = p.replace('test', 'sample_iter5')

        if not os.path.exists('datasets/final/veri_sample_iter5'):
            os.mkdir('datasets/final/veri_sample_iter5')

        if not os.path.exists(new_path):
            print('creating:' + os.path.dirname(new_path))
            os.mkdir(new_path)
        img = images[idx].cpu().detach().squeeze(0)
        if position and raw_pics:
            img = segmentation.reverse_segment(img, raw_pics[idx], position[idx])
        img = unloader(img)  # .convert('RGB')
        print("exporting to:" + str(os.path.join(new_path, f)))
        img.save(os.path.join(new_path, f))


def run_adv():
    attack_on_data = 'test'
    print('adversarial attacking: FingerSafe')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataroot = './datasets/final/veri_test'
    dataset = FingerprintAdv(dataroot, phase=attack_on_data)
    batch_size = 6

    adv_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False)

    model = Model(nclasses=268, classify=False).to(device)
    model.load_state_dict(torch.load("./best_models/clean_split_1009.pth"))

    for param in model.parameters():
        param.requires_grad = False

    epsilon = 8. / 255.
    # FingerSafe
    attacker = FingerSafe(model, device, eps=epsilon, alpha=epsilon / 10, steps=5, random_start=True)

    # attacking polyU
    start = time.time()
    for i, (data, _, path) in enumerate(adv_loader):
        adv_images = attacker(data)
        export(path, adv_images)
    end = time.time()
    print(end - start)
    exit(0)


def cal_orient(images):
    ridge = ridge_orient()
    mean_orient = []
    for img in images:
        orient = ridge(img.cuda())
        mean_orient.append(orient)
    mean_orient = [o.cpu().numpy() for o in mean_orient]
    mean_orient = np.mean(np.array(mean_orient), axis=0)  # 224*224
    return torch.from_numpy(mean_orient)


def find_target(x, target_set, model):
    target_emb = []
    target_images = []

    # find target for each image
    for i_image in range(len(x)):
        image = x[i_image]
        random_class = [random.randint(0, 267) for _ in range(8)]  # choose 8 targets from train set
        print('target class: ' + str(random_class))
        target_set_random = [target_set.x_data[i * 6: (i + 1) * 6] for i in random_class]

        target_emb_centers = []
        target_embs = []
        target_orients = []
        for iclass in target_set_random:
            target_image = torch.stack(iclass, 0)
            target_embedding = model(target_image.cuda())  # 6*1000
            target_embedding_center = torch.mean(target_embedding, dim=0)  # 1000
            target_emb_centers.append(target_embedding_center)
            target_embs.append(target_embedding)
            # calculate the orientation of target class
            target_orients.append(cal_orient(iclass))  # 224*224

        # find the most dissimilar target class for source image
        source_embedding = model(x.cuda())  # 6*1000
        source_orient = cal_orient(x)  # 224*224
        emb_distance = 0
        orient_distance = 0
        temb_index = 0
        torient_index = 0
        for index in range(len(target_emb_centers)):
            # representation
            t = target_emb_centers[index]  # 1*1000
            # t = t.repeat(10, 1)  # 6*1000
            t = t.repeat(6, 1)  # 6*1000
            i_distance = torch.norm((t - source_embedding), p=2)
            if i_distance > emb_distance:  # find the most dissimilar
                emb_distance = i_distance
                temb_index = index
            # orientation
            o = target_orients[index]  # 224*224
            diff = torch.abs(o - source_orient)
            i_distance = torch.sin(diff)
            i_distance = torch.mean(i_distance.abs())
            if i_distance > orient_distance:
                orient_distance = i_distance
                torient_index = index

        # randomly pick an image from class T
        fig_index = random.randint(0, 5)
        target_emb.append(target_embs[temb_index][fig_index].cpu().detach().numpy())  # 1*1000
        target_images.append(target_set.x_data[torient_index * 6 + fig_index].numpy())
    target_emb = torch.from_numpy(np.array(target_emb)).cuda()  # 6*1000
    target_images = torch.from_numpy(np.array(target_images)).cuda()
    return target_emb, target_images


class Gray(object):

    def __call__(self, tensor):  # tensor: 3 * w * h
        # TODO: make efficient
        _, w, h = tensor.shape
        R = tensor[0]
        G = tensor[1]
        B = tensor[2]
        tmp = 0.299 * R + 0.587 * G + 0.114 * B
        tensor = tmp
        tensor = tensor.view(1, w, h)
        return tensor


if __name__ == '__main__':
    run_adv()
