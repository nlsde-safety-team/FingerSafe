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


# steal some code from torchattacks
class FingerSafe(nn.Module):
    def __init__(self, model, device, eps=0.3, alpha=2 / 255, steps=40, random_start=False, inc=None, dense=None):
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

    def forward(self, images, masks=None):
        r"""
        forward fingersafe
        """
        images = images.clone().detach().to(self.device)
        original_embs = self.model(images)
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # todo: FingerSafe: to make adv representation close to target representation
            loss_rep = L_representation(outputs, original_embs)  # far from itself
            loss_orientation = L_orientation_target(self.ridge, images, adv_images, masks=masks)
            loss_contrast = L_contrast(self.contrast, images, adv_images)

            cost = -1 * loss_rep - 1e2 * loss_orientation + 0.01 * loss_contrast

            print(loss_rep, loss_orientation, loss_contrast)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
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


# calculate L_adv
def L_representation(features, target):
    diff = features - target
    scale_factor = 1  # torch.sqrt(torch.sum(torch.pow(target, 2), dim=1))
    distance = torch.sum(torch.pow(diff, 2), dim=1)
    distance = distance / scale_factor
    return torch.mean(distance)


def L_orientation_lsm(ridge, images, adv_images):
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


# calculate L_O
def L_orientation_target(ridge, t_images, adv_images, masks=None):
    # let orientation of adv_images close to target images
    distance = 0
    for i in range(len(t_images)):
        if masks is not None:  # mask in social media protection
            mask = masks[i].bool()
            images_orient = ridge(t_images[i]).masked_fill(~mask.to('cuda'), value=0)
            adv_images_orient = ridge(adv_images[i]).masked_fill(~mask.to('cuda'), value=0)
        else:
            images_orient = ridge(t_images[i])  # get orientation field of target image
            adv_images_orient = ridge(adv_images[i])  # get orientation field of adv image

        diff = torch.abs(adv_images_orient - images_orient)
        d = torch.sin(diff)
        d = d.abs()
        distance += torch.mean(d)
    return distance / (len(t_images))


# calculate L_C
def L_contrast(contrast, images, adv_images):
    distance = 0
    for i in range(len(images)):
        local1, _, sal1 = contrast(images[i])
        local2, _, sal2 = contrast(adv_images[i])
        sal1, sal2 = sal1.unsqueeze(0).repeat(3, 1, 1), sal2.unsqueeze(0).repeat(3, 1, 1)
        con1 = torch.mul(local1, sal2)
        con2 = torch.mul(local2, sal2)
        distance = distance + torch.mean(F.relu(con2 - con1))
    distance = distance / len(images)
    return distance


def L_color(images, adv_images):
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
