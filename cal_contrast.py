import numpy as np
import cv2
import torch
from convolutions import Conv2d
import torch.nn as nn
from torch.nn.parameter import Parameter


def show_img(img, name):
    # img = img.squeeze()
    # img = img.permute(1, 2, 0)
    img = img.cpu().detach().numpy()
    img = ((img / np.max(img)) * 255).astype('uint8')
    img = cv2.resize(img, (224, 224))
    cv2.imshow(name, img)


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


class cal_contrast(nn.Module):
    def __init__(self):
        super(cal_contrast, self).__init__()
        rc = 2  # follow table 4 in 2012 perceptual contrast paper
        rs = 4
        center_kernel = self.get_center(rc, rc * 6 + 1)
        # surround_kernel = self.get_center(rs, rs * 6 + 1)
        # center_kernel = self.get_surround(rc, rs, rc * 6 + 1)
        surround_kernel = self.get_surround(rc, rs, rs * 6 + 1)
        box_kernel = torch.ones(3, 3)
        box_kernel = box_kernel / torch.sum(box_kernel)  # normalize
        gaussian_kernel = self.get_gaussian(3, 9)
        self.gray = Gray()

        self.center_conv = Conv2d(in_channels=1, out_channels=1, kernel_size=center_kernel.shape[0]).cuda()
        self.surround_conv = Conv2d(in_channels=1, out_channels=1, kernel_size=surround_kernel.shape[0]).cuda()
        self.box_conv = Conv2d(in_channels=1, out_channels=1, kernel_size=box_kernel.shape[0]).cuda()
        self.gaussian_conv = Conv2d(in_channels=1, out_channels=1, kernel_size=gaussian_kernel.shape[0]).cuda()

        center_kernel = torch.FloatTensor(center_kernel).unsqueeze(0).unsqueeze(0)
        surround_kernel = torch.FloatTensor(surround_kernel).unsqueeze(0).unsqueeze(0)
        box_kernel = torch.FloatTensor(box_kernel).unsqueeze(0).unsqueeze(0)
        gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)

        self.center_conv.weight, self.surround_conv.weight, self.box_conv.weight, self.gaussian_conv.weight = Parameter(
            center_kernel.cuda(), requires_grad=True), \
                                                                                                              Parameter(
                                                                                                                  surround_kernel.cuda(),
                                                                                                                  requires_grad=True), Parameter(
            box_kernel.cuda(), requires_grad=True), Parameter(gaussian_kernel.cuda(), requires_grad=True)
        self.center_conv.bias, self.surround_conv.bias, self.box_conv.bias, self.gaussian_conv.bias = Parameter(
            torch.FloatTensor([0]).cuda(), requires_grad=False), \
                                                                                                      Parameter(
                                                                                                          torch.FloatTensor(
                                                                                                              [
                                                                                                                  0]).cuda(),
                                                                                                          requires_grad=False), \
                                                                                                      Parameter(
                                                                                                          torch.FloatTensor(
                                                                                                              [
                                                                                                                  0]).cuda(),
                                                                                                          requires_grad=False), \
                                                                                                      Parameter(
                                                                                                          torch.FloatTensor(
                                                                                                              [
                                                                                                                  0]).cuda(),
                                                                                                          requires_grad=False)

    def forward(self, img):
        img_b_1 = img[2, ...]
        img_g_1 = img[1, ...]
        img_r_1 = img[0, ...]
        saliency = self.saliency(self.gray(img).squeeze())
        # local estimate of contrast
        contrast_b_1, contrast_g_1, contrast_r_1 = self.get_contrast(img_b_1, img_g_1, img_r_1)
        '''
        cv2.imshow('contrast',
                   (255 * contrast_b_1 / torch.max(contrast_b_1)).squeeze().detach().cpu().numpy().astype('uint8'))
        cv2.waitKey(0)
        '''
        '''
        img_b_2, img_g_2, img_r_2 = self.antialiasing(img_b_1, img_g_1, img_r_1)
        img_b_2, img_g_2, img_r_2 = self.downsample(img_b_2, img_g_2, img_r_2)
        contrast_b_2, contrast_g_2, contrast_r_2 = self.get_contrast(img_b_2, img_g_2, img_r_2)

        img_b_3, img_g_3, img_r_3 = self.antialiasing(img_b_2, img_g_2, img_r_2)
        img_b_3, img_g_3, img_r_3 = self.downsample(img_b_3, img_g_3, img_r_3)
        contrast_b_3, contrast_g_3, contrast_r_3 = self.get_contrast(img_b_3, img_g_3, img_r_3)

        tau_b_1, tau_g_1, tau_r_1 = torch.var(img_b_1), torch.var(img_g_1), torch.var(img_r_1)
        tau_b_2, tau_g_2, tau_r_2 = torch.var(img_b_2), torch.var(img_g_2), torch.var(img_r_2)
        tau_b_3, tau_g_3, tau_r_3 = torch.var(img_b_3), torch.var(img_g_3), torch.var(img_r_3)

        C_b = tau_b_1 * torch.mean(contrast_b_1) + tau_b_2 * torch.mean(contrast_b_2) + tau_b_3 * torch.mean(
            contrast_b_3)
        C_g = tau_g_1 * torch.mean(contrast_g_1) + tau_g_2 * torch.mean(contrast_g_2) + tau_g_3 * torch.mean(
            contrast_g_3)
        C_r = tau_r_1 * torch.mean(contrast_r_1) + tau_r_2 * torch.mean(contrast_r_2) + tau_r_3 * torch.mean(
            contrast_r_3)

        C_b, C_g, C_r = C_b / 3, C_g / 3, C_r / 3  # by level pf pyramids

        C_WLF = (C_b + C_g + C_r) / 3  # by color, C_WLF is global estimate
        '''
        # contrast_img = torch.cat([contrast_b_1.unsqueeze(0), contrast_g_1.unsqueeze(0), contrast_r_1.unsqueeze(0)],
        #                          dim=0).squeeze()
        contrast_img = torch.cat([contrast_r_1.unsqueeze(0), contrast_g_1.unsqueeze(0), contrast_b_1.unsqueeze(0)],
                                 dim=0).squeeze()
        # print(contrast_img.shape)

        # local contrast: contrast_b_1, contrast_b_2, contrast_b_3
        # global contrast: C_WLF

        return contrast_img, 0, saliency

    def saliency(self, img):
        # img = img.permute(1, 2, 0)
        # SpectralResidual  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.125.5641&rep=rep1&type=pdf
        img = img * 255
        img_fft = torch.fft.fft2(img)
        fft_mag = torch.abs(img_fft)

        img_fft = torch.view_as_real(img_fft)
        img_fft_final = img_fft.detach().clone()

        fft_boxconv = torch.clip(self.box_conv(torch.log(fft_mag.unsqueeze(0).unsqueeze(0))), 0, 255).squeeze()
        spectral_residual = torch.exp(torch.log(fft_mag) - fft_boxconv)

        img_fft_final[..., 0] = torch.div(torch.mul(img_fft[..., 0], spectral_residual), fft_mag)

        img_fft_final[..., 1] = torch.div(torch.mul(img_fft[..., 1], spectral_residual), fft_mag)
        img_fft = torch.view_as_complex(img_fft_final)

        saliency = torch.fft.ifft2(img_fft)
        saliency = torch.view_as_real(saliency)
        saliency_final = saliency.detach().clone().cuda()

        saliency_final = torch.pow(saliency[..., 0], 2) + torch.pow(saliency[..., 1], 2)
        saliency = self.gaussian_conv(saliency_final.unsqueeze(0).unsqueeze(0).cuda()).squeeze()
        saliency = (saliency - torch.min(saliency)) / (torch.max(saliency) - torch.min(saliency))

        # saliency = (saliency.detach().cpu().numpy() * 255).astype('uint8')
        # cv2.imshow("Output", saliency)
        # cv2.waitKey(0)
        return saliency

    def downsample(self, img_b, img_g, img_r):
        return img_b[0:-1:2, 0:-1:2], img_g[0:-1:2, 0:-1:2], img_r[0:-1:2, 0:-1:2]

    def get_contrast(self, img_b, img_g, img_r):
        return self.get_DOG(img_b), self.get_DOG(img_g), self.get_DOG(img_r)

    def get_DOG(self, img):
        img = img.unsqueeze(0).unsqueeze(0)
        Rc = self.center_conv(img)
        Rs = self.surround_conv(img)
        c_TT = torch.div((Rc - Rs), Rc).squeeze()
        return c_TT

    def filter(self, img):
        img_fft = torch.fft.fft2(img)
        size = img.shape[0]
        downsampl_size = int(size / 2)
        img_fft[downsampl_size:size, downsampl_size:size] = 0
        img_fft[0: downsampl_size, downsampl_size:size] = 0
        img_fft[downsampl_size:size, 0: downsampl_size] = 0
        img_filtered = torch.fft.ifft2(img_fft)
        return img_filtered

    def antialiasing(self, img_b, img_g, img_r):
        return torch.abs(self.filter(img_b)), torch.abs(self.filter(img_g)), torch.abs(self.filter(img_r))

    def get_center(self, rc, size):
        kernel = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                kernel[i, j] = np.exp(- (np.abs(i-np.floor(size/2))/rc)**2 - (np.abs(j-np.floor(size/2))/rc)**2)
        return kernel

    def get_surround(self, rc, rs, size):
        kernel = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                kernel[i, j] = np.exp(- (np.abs(i-np.floor(size/2))/rs)**2 - (np.abs(j-np.floor(size/2))/rs)**2)
        kernel = kernel * 0.85 * ((rc/rs) ** 2)
        return kernel


    def get_gaussian(self, sigma, size):
        return cv2.getGaussianKernel(size, sigma) * cv2.getGaussianKernel(size, sigma).T
