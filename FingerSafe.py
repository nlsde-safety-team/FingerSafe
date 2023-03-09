import torch
import torch.nn as nn
import lpips
from L_orientation import ridge_orient
import numpy as np
import cv2
from cal_contrast import cal_contrast
import torch
import pickle
import torch.nn.functional as F
from models.inception_resnet_v1 import ResNet
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as functional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from torchvision import transforms
import os
from criterion import Criterion
from L_orientation import Gray
from utils import draw_convergence
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# steal some code from torchattacks
class FingerSafe(nn.Module):
    def __init__(self, model, device, eps=0.3, alpha=2 / 255, steps=40, random_start=False, inc=None, dense=None, lamda=100, gamma=500, gauss_1=7, gauss_sigma=3, draw_convergence=False):
        super(FingerSafe, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.ridge = ridge_orient()
        self.contrast = cal_contrast()
        self.pad = torch.nn.ZeroPad2d((31, 31, 31, 31))
        self.model = model
        self.criterion = Criterion()
        self.gray = Gray()
        self.device = device
        self.image_per_class = 6
        # self.target_feature = target_feature  # target feature
        # self.target_image = target_image
        self.inception = inc
        self.densenet = dense
        self.lamda = lamda
        self.gamma = gamma
        self.gauss_1 = gauss_1
        self.gauss_sigma = gauss_sigma
        self.draw_convergence = draw_convergence

    def forward_untarget(self, images):
        print(images.shape)
        images = images.clone().squeeze().detach().to(self.device)
        adv_images = images.clone().detach()

        for i in range(self.steps):
            images = images.clone().detach().to(self.device)
            adv_images = adv_images.clone().detach()
            images.requires_grad = True
            adv_images.requires_grad = True

            adv_images_rep = self.model(adv_images)
            images_rep = self.model(images)
            pred = images_rep.data.max(1)[1]

            adv_images_rep_dim0 = adv_images_rep.repeat(self.image_per_class, 1, 1)
            adv_images_rep_dim1 = adv_images_rep.repeat(self.image_per_class, 1, 1).transpose(0, 1)
            images_rep_dim1 = images_rep.repeat(self.image_per_class, 1, 1).transpose(0, 1)

            mask = (torch.ones(self.image_per_class, self.image_per_class) - torch.eye(self.image_per_class)).to(self.device)

            # loss_rep = torch.sum((adv_images_rep_dim0 - images_rep_dim1).abs(), dim=-1)  # L1
            # loss_rep_diff = torch.sum((adv_images_rep_dim0 - adv_images_rep_dim1).abs(), dim=-1).to(
            #     self.device)  # L1

            loss_rep = torch.norm((adv_images_rep_dim0 - images_rep_dim1), dim=-1, p=2)  # L2
            loss_rep_diff = torch.norm((adv_images_rep_dim0 - adv_images_rep_dim1), dim=-1, p=2).to(
                self.device)  # L2

            loss_rep_diff = torch.einsum('ij, ij->ij', loss_rep_diff, mask)
            # loss_rep_diff = F.pairwise_distance(adv_images_rep_dim0, adv_images_rep_dim1, p=2)
            # print(loss_rep.shape)
            # print(loss_rep_diff.shape)  # expect 6*6

            loss_orientation = L_orientation_lsm(self.ridge, images, adv_images)
            loss_orientation_diff = L_orientation_lsm(self.ridge, adv_images, adv_images).to(self.device)
            loss_orientation_diff = torch.einsum('ij, ij->ij', loss_orientation_diff, mask)

            loss_contrast = L_contrast(self.contrast, images, adv_images)
            L_V = torch.mean(loss_rep) + torch.sum(loss_rep_diff) / (self.image_per_class * (self.image_per_class-1))
            L_O = torch.mean(loss_orientation) + torch.sum(loss_orientation_diff) / (self.image_per_class * (self.image_per_class-1))
            L_C = loss_contrast
            print(L_V, L_O, L_C)
            # cost = -10 * L_V - 1e2 * L_O + 5e-2 * L_C  # 0912B
            # cost = -1 * L_V - 10 * L_O + 5e-3 * L_C  # 0920a
            # cost = -1 * L_V - 10 * L_O + 5e-2 * L_C  # 0920b
            # cost = -1 * L_V - 10 * L_O + 5e-1 * L_C  # 0920c
            # cost = -1 * L_V - 1 * L_O + 5e-1 * L_C  # 0920d
            # cost = -1 * L_V - 10 * L_O + 1 * L_C  # 0920e
            # cost = -1 * L_V - 1 * L_O + 5e-2 * L_C  # 0923f
            # cost = -1 * L_V  # 0923g
            # cost = -1 * L_V - 1 * L_O + 1e-2 * L_C  # 0923h
            # cost = -1 * L_V - 1 * L_O + 5e-3 * L_C  # 0923i
            # cost = -1 * L_V - 0.1 * L_O + 1e-3 * L_C  # 0923j
            # cost = -1 * L_V - 10 * L_O + 0.1 * L_C  # 0923k
            # cost = -1 * L_V - 1 * L_O + 5e-5 * L_C  # color
            # cost = -1 * L_V - 10 * L_O + 0 * L_C  # noLc
            # cost = -1 * L_V - 10 * L_O + 0.05 * L_C  # 1001m
            cost = -1 * L_V - 10 * L_O + 0.1 * L_C  # 1001n now
            # cost = -1 * L_V - 0 * L_O + 0.1 * L_C  # noLF

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)
            grad = grad[0]

            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).clone().detach()
        return adv_images

    def forward(self, images, masks=None):
        images = images.clone().detach().to(self.device)
        original_embs = self.model(images)
        adv_images = images.clone().detach()
        costs = []

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss_rep = L_representation(outputs, original_embs)  # far from itself
            loss_orientation = L_orientation_target(self.ridge, images, adv_images, masks=masks)
            loss_contrast = L_contrast(self.contrast, images, adv_images)
            cost = -1 * loss_rep - self.lamda * loss_orientation + self.gamma * loss_contrast
            costs.append(cost.detach().cpu())
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            # todo here to mask grad, only for physical world
            # masked_grad = grad.masked_fill(masks.to(self.device), value=0)
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        if self.draw_convergence:
            costs = np.array(costs)
            np.save('./convergence/fingersafe.npy', costs)
            exit(0)
        return adv_images

    def forward_lowkey(self, images, masks=None, lowkey_lpips = 5.0):
        """
        LowKey
        """
        images = images.clone().detach().to(self.device)
        original_resnet = self.model(images)
        original_inception = self.inception(images)
        original_densenet = self.densenet(images)

        adv_images = images.clone().detach().to(self.device)
        gauss = torchvision.transforms.GaussianBlur(self.gauss_1, sigma=self.gauss_sigma)
        loss_fn_resnet = lpips.LPIPS(net='alex').to(self.device)
        costs = []

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            # resnet
            outputs_resnet = self.model(adv_images)
            guass_outputs_resnet = self.model(gauss(adv_images))
            # inception
            outputs_inception = self.inception(adv_images)
            guass_outputs_inception = self.inception(gauss(adv_images))
            # densenet
            outputs_densenet = self.densenet(adv_images)
            guass_outputs_densenet = self.densenet(gauss(adv_images))

            # todo: FingerSafe: to make adv representation close to target representation
            loss_resnet = L_representation(outputs_resnet, original_resnet) + L_representation(guass_outputs_resnet, original_resnet)
            loss_resnet = loss_resnet / torch.sqrt(L_representation(original_resnet, 0))

            loss_inception = L_representation(outputs_inception, original_inception) + L_representation(guass_outputs_inception,
                                                                                               original_inception)
            loss_inception = loss_inception / torch.sqrt(L_representation(original_inception, 0))

            loss_densenet = L_representation(outputs_densenet, original_densenet) + L_representation(guass_outputs_densenet,
                                                                                               original_densenet)
            loss_densenet = loss_densenet / torch.sqrt(L_representation(original_densenet, 0))

            loss_rep = (loss_resnet + loss_inception + loss_densenet) / 6

            loss_lpips = loss_fn_resnet(images, adv_images)
            cost = -1 * loss_rep + lowkey_lpips * torch.mean(loss_lpips)  # orient: A=1, B=10, C=100, D=1000
            costs.append(cost.detach().cpu())

            # print(loss_rep, torch.mean(loss_lpips))

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.alpha * grad.sign()
            # delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            delta = adv_images - images
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        if self.draw_convergence:
            costs = np.array(costs)
            np.save('./convergence/lowkey.npy', costs)
            exit(0)
        return adv_images


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def L_color(images, adv_images):
    # pairwise distance, 1 if similiar, 0 if dissimiliar
    distance = 0
    for i in range(len(images)):
        delta = adv_images[i] - images[i]
        # noise in BGR
        R_bar = (adv_images[i, 2, ...] + images[i, 2, ...]) / 2
        color_distance = torch.sqrt((2 + R_bar) * torch.pow(delta[2, ...] * 255, 2) + 4 * torch.pow(delta[1, ...] * 255, 2) + (2 + (1-R_bar) * torch.pow(delta[0, ...] * 255, 2)))
        distance += torch.norm(color_distance, p='fro')
    return distance / len(images)

def L_representation(features, target):
    # d = F.pairwise_distance(features, target, p=2)
    diff = features - target
    scale_factor = 1  # torch.sqrt(torch.sum(torch.pow(target, 2), dim=1))
    distance = torch.sum(torch.pow(diff, 2), dim=1)
    distance = distance / scale_factor
    return torch.mean(distance)

def L_orientation_lsm(ridge, images, adv_images):
    # pairwise distance, 1 if similiar, 0 if dissimiliar
    # distance = 0
    loss_matrix = torch.zeros(6, 6)
    for i in range(len(images)):
        for j in range(len(images)):
            images_orient = ridge(images[i])
            adv_images_orient = ridge(adv_images[j])

            diff = torch.abs(adv_images_orient - images_orient)
            # d = F.pairwise_distance(images_orient, adv_images_orient, p=2)
            d = torch.sin(diff)
            d = d.abs()
            loss_matrix[i, j] = torch.mean(d)
            # distance += torch.mean(d)  # + torch.std(d)
    return loss_matrix

def L_orientation_target(ridge, t_images, adv_images, masks=None):
    # let orientation of adv_images close to target images
    distance = 0
    for i in range(len(t_images)):
        if masks is not None:  # physical world
            mask = masks[i].bool()
            images_orient = ridge(t_images[i]).masked_fill(~mask.to('cuda'), value=0)
            adv_images_orient = ridge(adv_images[i]).masked_fill(~mask.to('cuda'), value=0)
        else:
            images_orient = ridge(t_images[i]) 
            adv_images_orient = ridge(adv_images[i]) 

        # images_orient = ridge(t_images[i])
        # adv_images_orient = ridge(adv_images[i])

        diff = torch.abs(adv_images_orient - images_orient)
        d = torch.sin(diff)
        d = d.abs()
        distance += torch.mean(d)
    return distance / (len(t_images))


def L_contrast(contrast, images, adv_images):
    # pairwise distance, 1 if similiar, 0 if dissimiliar
    distance = 0
    for i in range(len(images)):
        # delta = (adv_images[i] - images[i]) * 255
        # print(delta.shape)
        # exit(0)
        # noise in BGR
        # contrast_img, C_WLF = contrast(delta)
        # con1, _, sal1 = contrast(adv_images[i])
        # con2, _, sal2 = contrast(images[i])
        # contrast_img = con1 - con2
        # distance += torch.norm(contrast_img, p='fro')
        local1, _, sal1 = contrast(images[i])
        local2, _, sal2 = contrast(adv_images[i])
        sal1, sal2 = sal1.unsqueeze(0).repeat(3, 1, 1), sal2.unsqueeze(0).repeat(3, 1, 1)
        con1 = torch.mul(local1, sal2)
        con2 = torch.mul(local2, sal2)
        # distance = distance + torch.sum(F.relu(con2 - con1))  # 1116: sum to mean
        distance = distance + torch.mean(F.relu(con2 - con1))
    distance = distance / len(images)
    return distance



def L_color(images, adv_images):
    # pairwise distance, 1 if similiar, 0 if dissimiliar
    distance = 0
    for i in range(len(images)):
        delta = adv_images[i] - images[i]
        # noise in BGR
        R_bar = (adv_images[i, 2, ...] + images[i, 2, ...]) / 2
        color_distance = torch.sqrt(
            (2 + R_bar) * torch.pow(delta[2, ...] * 255, 2) + 4 * torch.pow(delta[1, ...] * 255, 2) + (
                    2 + (1 - R_bar) * torch.pow(delta[0, ...] * 255, 2)))
        distance += torch.norm(color_distance, p='fro')
    return distance / len(images)


def show_img(img, name):
    img = img.cpu().detach().numpy()
    img = ((img / np.max(img + 1e-6)) * 255).astype('uint8')
    img = cv2.resize(img, (224, 224))
    cv2.imshow(name, img)


def export(paths, images):
    unloader = torchvision.transforms.ToPILImage()
    rgb2bgr = [2, 1, 0]
    images = images[:, rgb2bgr, :, :]

    for idx in range(len(images)):
        p, f = os.path.split(paths[0][idx])
        if p.find('train') != -1:
            new_path = p.replace('train', 'perturb2')
        else:
            new_path = p.replace('veri_test', 'sample2')
        os.makedirs(os.path.dirname(os.path.join(new_path, f)), exist_ok=True)
        img = unloader(images[idx].cpu().detach().squeeze(0)).convert('RGB')
        print("exporting to:" + str(os.path.join(new_path, f)))
        img.save(os.path.join(new_path, f))


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
