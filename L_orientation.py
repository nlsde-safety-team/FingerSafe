import numpy as np
import cv2
import torch
from convolutions import Conv2d
# from torch.nn import Conv2d


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


class ridge_orient(object):

    def __call__(self, _normim):
        gray = Gray()
        _normim = gray(_normim).squeeze()
        gradient_sigma = 1
        block_sigma = 5
        orient_smooth_sigma = None
        # Calculate image gradients.
        sze = np.fix(6 * gradient_sigma)
        if np.remainder(sze, 2) == 0:
            sze = sze + 1

        gauss = cv2.getGaussianKernel(np.int(sze), gradient_sigma)
        f = gauss * gauss.T

        fy, fx = np.gradient(f)  # Gradient of Gaussian

        # Gx = signal.convolve2d(self._normim, fx, mode='same')
        # Gy = signal.convolve2d(self._normim, fy, mode='same')
        from torch.nn.parameter import Parameter
        Cx = Conv2d(in_channels=1, out_channels=1, kernel_size=fx.shape[0]).cuda()
        Cy = Conv2d(in_channels=1, out_channels=1, kernel_size=fy.shape[0]).cuda()
        fx = torch.FloatTensor(fx).unsqueeze(0).unsqueeze(0)
        fy = torch.FloatTensor(fy).unsqueeze(0).unsqueeze(0)
        _normim = _normim.unsqueeze(0).unsqueeze(0)
        Cx.weight, Cy.weight = Parameter(fx.cuda(), requires_grad=False), Parameter(fy.cuda(), requires_grad=False)
        Cx.bias, Cy.bias = Parameter(torch.FloatTensor([0]).cuda(), requires_grad=False), Parameter(
            torch.FloatTensor([0]).cuda(), requires_grad=False)
        Gx = Cx(_normim)
        Gy = Cy(_normim)
        _normim = _normim.squeeze()

        # Gxx = np.power(Gx,2)
        # Gyy = np.power(Gy,2)
        Gxx = torch.pow(Gx, 2)
        Gyy = torch.pow(Gy, 2)
        Gxy = Gx * Gy

        # Now smooth the covariance data to perform a weighted summation of the data.
        sze = np.fix(6 * block_sigma)
        if np.remainder(sze, 2) == 0:
            sze = sze + 1

        gauss = cv2.getGaussianKernel(np.int(sze), block_sigma)
        f = gauss * gauss.T

        C = Conv2d(in_channels=1, out_channels=1, kernel_size=f.shape[0]).cuda()
        f = torch.FloatTensor(f).unsqueeze(0).unsqueeze(0)
        C.weight = Parameter(f.cuda(), requires_grad=False)
        C.bias = Parameter(torch.FloatTensor([0]).cuda(), requires_grad=False)
        Gxx = C(Gxx)
        Gyy = C(Gyy)
        Gxy = 2 * C(Gxy)

        # Analytic solution of principal direction
        # denom = np.sqrt(np.power(Gxy,2) + np.power((Gxx - Gyy),2)) + np.finfo(float).eps
        denom = torch.sqrt(torch.pow(Gxy, 2) + torch.pow((Gxx - Gyy), 2)) + torch.tensor(np.finfo(float).eps)

        sin2theta = Gxy / denom  # Sine and cosine of doubled angles
        cos2theta = (Gxx - Gyy) / denom

        if orient_smooth_sigma:
            sze = np.fix(6 * orient_smooth_sigma)
            if np.remainder(sze, 2) == 0:
                sze = sze + 1
            gauss = cv2.getGaussianKernel(np.int(sze), orient_smooth_sigma)
            f = gauss * gauss.T
            C = Conv2d(in_channels=1, out_channels=1, kernel_size=f.shape[0]).cuda()
            f = torch.FloatTensor(f).unsqueeze(0).unsqueeze(0)
            C.weight = Parameter(f.cuda(), requires_grad=False)
            C.bias = Parameter(torch.FloatTensor([0]).cuda(), requires_grad=False)
        cos2theta = C(cos2theta).squeeze()
        sin2theta = C(sin2theta).squeeze()

        # self._orientim = np.pi/2 + np.arctan2(sin2theta,cos2theta)/2
        _orientim = torch.tensor(np.pi) / 2 + torch.atan2(sin2theta, cos2theta) / 2

        return _orientim


