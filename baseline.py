import torch, torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.inception_resnet_v1 import ResNet as Model
from FingerprintDataset import FingerprintAdv
import os
import argparse
import cv2
import time
import foolbox
import torchattacks


def export(paths, images):
    unloader = torchvision.transforms.ToPILImage()
    for idx in range(len(images)):
        p, f = os.path.split(paths[0][idx])
        if p.find('train') != -1:
            new_path = p.replace('train', 'perturb_new_MHS_boundary')
        else:
            new_path = p.replace('test', 'sample_pgd')
        os.makedirs(os.path.dirname(os.path.join(new_path, f)), exist_ok=True)
        img = unloader(images[idx].cpu().detach().squeeze(0))
        img.save(os.path.join(new_path, f))
        print("exporting to: {}".format(os.path.join(new_path, f)))


def run_adv(epoch=-1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int,
                        help='batch size', default=6)
    parser.add_argument('--dataroot', '-d', type=str,
                        help='root of loading data',
                        default="./datasets/final/veri_test")
    parser.add_argument('--epsilon', '-e', type=float,
                        default=8. / 255.)
    args = parser.parse_args()

    dataroot = args.dataroot
    batch_size = args.batch_size
    epsilon = args.epsilon
    weights_name = './best_models/clean_split_1009.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FingerprintAdv(dataroot, phase='test')
    adv_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False)
    model = Model(nclasses=268, classify=True).to(device)
    model.load_state_dict(torch.load(weights_name))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # todo: PGD
    attacker = torchattacks.PGD(model, eps=epsilon, alpha=epsilon / 2, steps=20)

    # todo BoundaryAttack
    # attacker = foolbox.attacks.BoundaryAttack(init_attack=None, steps=50, spherical_step=0.0001, source_step=0.0001,
    #                                           source_step_convergance=1e-07, step_adaptation=1.5, tensorboard=False,
    #                                           update_stats_every_k=1)
    # fmodel = foolbox.PyTorchModel(model, bounds=(0, 1))

    # attacking
    start = time.time()
    for i, (data, label, path) in enumerate(adv_loader):
        print(i)
        # todo PGD
        adv_images = attacker(data, label)
        export(path, adv_images)
        # todo BoundaryAttack
        # raw, clipped, is_adv = attacker(fmodel, data.cuda(), label.cuda(), epsilons=None)
        # export(path, clipped)
        # print(is_adv)

    end = time.time()
    print("time: " + str(end - start))
    exit(0)



def guass_noise(img):
    mid, sigma = torch.mean(img), torch.std(img)
    epsilon = 8./255.
    # noise = epsilon * np.random.normal(mid.numpy(), sigma.numpy(), img.shape)
    noise = np.random.normal(0.0, epsilon, img.shape)
    img = img + torch.from_numpy(noise)
    return img


def find_mask(img, gradcam=None):
    # preprocess fingerprint as mask
    # out = fingerprint_enhancer.enhance_Fingerprint(img)

    # grad cam as mask
    mask, _ = gradcam(img)
    return mask


def IJCB2015(img):
    img = np.array(img)
    img_medianBlur = cv2.medianBlur(img, 3)
    img_His = cv2.equalizeHist(img_medianBlur)
    img_GaussBlur = cv2.GaussianBlur(img_His, (9, 9), 2)
    img_sharpen = img_His - img_GaussBlur
    # cv2.imshow('IJCB2015', img_sharpen)
    return img_sharpen


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
