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
import argparse
# https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def export(paths, images, position=None, raw_pics=None, output='sample_fingersafe_gamma_10000'):
    unloader = torchvision.transforms.ToPILImage()
    # rgb2bgr = [2, 1, 0]
    # images = images[:, rgb2bgr, :, :]
    for idx in range(len(images)):
        p, f = os.path.split(paths[0][idx])
        if p.find('train') != -1:
            new_path = p.replace('train', 'perturb_tmp_I')
        else:
            new_path = p.replace('test', output)

        # if not os.path.exists('datasets/fingerprint_verification/perturb_fingeradv_0912C'):
        #     os.mkdir('datasets/fingerprint_verification/perturb_fingeradv_0912C')

        if not os.path.exists('datasets/final/veri_' + output):
            os.mkdir('datasets/final/veri_' + output)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int,
                        help='batch size', default=6)
    parser.add_argument('--dataroot', '-d', type=str,
                        help='root of loading data',
                        default="./datasets/final/veri_test")
    parser.add_argument('--mask', '-m', type=str,
                        help='root of mask of fingers in physical world experiment',
                        default="./datasets/physical_square/masks/evaluation")
    parser.add_argument('--epsilon', '-e', type=float,
                        default=8./255.)
    parser.add_argument('--output', type=str, default='sample_fingersafe_gamma_10000')
    parser.add_argument('--lowkey_lpips', type=float, default=5.0)
    parser.add_argument('--method', type=str, default='fingersafe')
    parser.add_argument('--gamma', type=float, default=500)
    parser.add_argument('--gauss', type=int, default=7)
    parser.add_argument('--sigma', type=int, default=3)
    parser.add_argument('--victim', type=str, default='ResNet')
    args = parser.parse_args()

    draw = False

    attack_on_data = 'test'
    if args.method == 'fingersafe':
        print('adversarial attacking: FingerSafe, gamma: {}, victim: {}'.format(args.gamma, args.victim))
    else:
        print('adversarial attacking: Lowkey, lpips: {}, gauss: {}, sigma: {}'.format(args.lowkey_lpips, args.gauss, args.sigma))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    dataroot = args.dataroot
    dataset = FingerprintAdv(dataroot, phase=attack_on_data)
    # todo：physical word
    # root_mask = args.mask
    # dataset = Fingerprint_Mask(dataroot, root_mask, phase=attack_on_data)

    adv_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False)

    if args.victim == 'ResNet':
        model = Model(nclasses=268, classify=False).to(device)
        model.load_state_dict(torch.load("./best_models/clean_split_1009.pth"))
    elif args.victim == 'DenseNet':
        model = DenseNet(nclasses=268, classify=False).to(device)
        model.load_state_dict(torch.load("./best_models/densenet_clean_best.pth"))
    else:
        model = Inceptionv3(nclasses=268, classify=False).to(device)
        model.load_state_dict(torch.load("./best_models/inception_clean_best.pth"))

    # todo Lowkey
    if args.method != 'fingersafe':
        inception = Inceptionv3(nclasses=268, classify=False).to(device)
        inception.load_state_dict(torch.load("./best_models/inception_clean_best.pth"))
        densenet = DenseNet(nclasses=268, classify=False).to(device)
        densenet.load_state_dict(torch.load("./best_models/densenet_clean_best.pth"))

    for param in model.parameters():
        param.requires_grad = False

    epsilon = args.epsilon
    if args.method == 'fingersafe':
        attacker = FingerSafe(model, device, eps=epsilon, alpha=epsilon / 10, steps=20, random_start=True, gamma=args.gamma, draw_convergence=draw)
    else:
        attacker = FingerSafe(model, device, eps=epsilon, alpha=0.0025, steps=20, random_start=True, inc=inception, dense=densenet, gauss_1=args.gauss, gauss_sigma=args.sigma, draw_convergence=draw)

    start = time.time()
    for i, (data, _, path) in enumerate(adv_loader):
        if args.method == 'fingersafe':
            adv_images = attacker(data)
        else:
            adv_images = attacker.forward_lowkey(data, lowkey_lpips=args.lowkey_lpips)
        export(path, adv_images, output=args.output)
    end = time.time()
    print(end - start)
    exit(0)

    # todo attacking physical

    # for i, (data, _, path, masks) in enumerate(adv_loader):
    #     adv_images = attacker(data, masks)
        # export(path, adv_images)

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
