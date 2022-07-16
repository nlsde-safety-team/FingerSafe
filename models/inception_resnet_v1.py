import os
# import requests
# from requests.adapters import HTTPAdapter

import torch
from torch import nn
from torch.nn import functional as F

from .utils.download import download_url_to_file
import cv2
import numpy as np


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')


        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.logits = nn.Linear(512, tmp_classes)
            load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x

from kymatio import Scattering2D
class ScattCNN(nn.Module):
    def __init__(self, input_shape=(225, 350), J=2, L=8, nclasses=10, device='cuda', classify=None):
        super(ScattCNN, self).__init__()

        self.scattering = Scattering2D(J=J, shape=input_shape)
        if torch.cuda.is_available():
            print("Move scattering to GPU")
            self.scattering = self.scattering.cuda()
        self.K = 1 + J * L + L ** 2 * (J - 1) * J // 2
        self.scatt_output_shape = tuple([x // 2 ** J for x in input_shape])
        self.bn = nn.BatchNorm2d(self.K)
        self.vgg = VGGNet().to(device)
        self.resnet = ResNet(nclasses, classify).to(device)

    def forward(self, x):
        # x = self.scattering(x)
        # x = x.view(-1, self.K, *self.scatt_output_shape)
        # x = self.bn(x)

        # Backbone
        x = self.resnet(x)
        return x
import torchvision.models as models
from models import Fingerprint_enhancer_our
class ResNet(nn.Module):
    def __init__(self, nclasses=268, classify=None):
        super(ResNet, self).__init__()
        net = models.resnet50(pretrained=True)
        net.classifier = nn.Sequential()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # in_channels = 1
        self.features = net
        # classifier of classification model
        self.classifier = nn.Sequential(
                nn.Linear(1000, 500),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(500, nclasses)
        )
        self.last_linear = nn.Linear(1000, 500, bias=False)
        self.last_bn = nn.BatchNorm1d(500, eps=0.001, momentum=0.1, affine=True)
        self.logits = nn.Linear(500, nclasses)
        self.classify = classify

    def forward(self, x):
        # MHS first
        # x = self.preprocess_batch(x)

        x = self.features(x)
        # like classification
        if self.classify:
            x = self.classifier(x.view(x.size(0), -1))
        else:
            x = F.normalize(x, p=2, dim=1)
        # x = self.last_linear(x)
        # x = self.last_bn(x)
        # if self.classify:
        #     x = self.logits(x)
        # else:
        #     x = F.normalize(x, p=2, dim=1)
        return x

    def preprocess_batch(self, x):
        # x: B*3*224*224
        out = []
        for i in range(len(x)):
            image = x[i].squeeze()  # 3*224*224 float tensor
            MHS_image = self.preprocess(image)  # 224*224 float array
            out.append(MHS_image[np.newaxis, :, :])  # 1*224*224 float array
        out = torch.tensor(out, dtype=torch.float32).cuda()  # B*3*224*224 float32 tensor
        return out

    def preprocess(self, image):
        # input image: 3*224*224 float tensor
        # RGB to Gray
        gray = Gray()
        image = gray(image).cpu().numpy()  # 224 * 224 float array
        # float to uint8
        image = image * 255
        image = image.astype(np.uint8)  # 224*224 uint8 array
        # MHS
        MHS_image = self.IJCB2015(image)  # 224*224 uint8 array
        return MHS_image / 255.  # 224*224 float array




    def IJCB2015(self, image):

        # 中值滤波
        img_medianBlur = cv2.medianBlur(image, 3)
        # 直方图均衡
        img_His = cv2.equalizeHist(img_medianBlur)
        # 锐化，减去高斯模糊
        img_GaussBlur = cv2.GaussianBlur(img_His, (9, 9), 2)
        img_sharpen = img_His - img_GaussBlur
        return img_sharpen

class Gray(object):

    def __call__(self, tensor):  # tensor: 3 * w * h
        # TODO: make efficient
        _, w, h = tensor.shape
        R = tensor[0]
        G = tensor[1]
        B = tensor[2]
        tmp = 0.299 * R + 0.587 * G + 0.114 * B
        tensor = tmp  # 224*224
        tensor = tensor.view(1, w, h)  # 1*224*224
        return tmp


class DenseNet(nn.Module):
    def __init__(self, nclasses=268, classify=None):
        super(DenseNet, self).__init__()
        net = models.densenet121(pretrained=True)
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Sequential()
        # net.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # in_channels = 1
        self.features = net
        # self.classifier = nn.Linear(num_ftrs, nclasses)
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(500, nclasses)
        )
        self.classify = classify

    def forward(self, x):
        x = self.features(x)
        # like classification
        if self.classify:
            x = self.classifier(x.view(x.size(0), -1))
        else:
            x = F.normalize(x, p=2, dim=1)
        return x


# class Inceptionv3(nn.Module):
#     def __init__(self, nclasses=112, classify=None):
#         super(Inceptionv3, self).__init__()
#         model_ft = models.inception_v3(aux_logits=False, pretrained=True)
#         num_ftrs = model_ft.fc.in_features
#         # model_ft.fc = nn.Linear(num_ftrs, nclasses)
#         model_ft.fc = nn.Sequential(
#             nn.Linear(num_ftrs, 500),
#             nn.ReLU(True),
#             nn.Dropout(0.5),
#             nn.Linear(500, nclasses)
#         )
#         self.net = model_ft
#         self.classify = classify
#
#     def forward(self, x):
#         x = self.net(x)
#         return x
class Inceptionv3(nn.Module):
    def __init__(self, nclasses=268, classify=None):
        super(Inceptionv3, self).__init__()
        net = models.inception_v3(pretrained=True)
        net.aux_logits = False
        net.AuxLogits = None
        num_ftrs = net.fc.in_features
        net.fc = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(500, nclasses)
        )
        self.classify = classify

    def forward(self, x):
        x = self.features(x)
        # like classification
        if self.classify:
            x = self.classifier(x.view(x.size(0), -1))
        else:
            x = F.normalize(x, p=2, dim=1)
        return x


class VGGNet(nn.Module):
    def __init__(self, num_classes=336):
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=False)   # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空
        self.features = net
        self.classifier = nn.Sequential(    # 定义自己的分类层
                nn.Linear(512 * 7 * 7, 1024),  # 512 * 7 * 7不能改变 ，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Siamese(nn.Module):
    def __init__(self, nclasses=268, device='cuda'):
        super(Siamese, self).__init__()

        # self.scattering = Scattering2D(J=J, shape=input_shape)
        # if torch.cuda.is_available():
        #     print("Move scattering to GPU")
        #     self.scattering = self.scattering.cuda()
        # self.K = 1 + J * L + L ** 2 * (J - 1) * J // 2
        # self.scatt_output_shape = tuple([x // 2 ** J for x in input_shape])
        # self.bn = nn.BatchNorm2d(self.K)
        # self.vgg = VGGNet().to(device)
        self.resnet = ResNet(nclasses, classify=False).to(device)
        # self.custom_cnn = CNN_github().to(device)

    def forward(self, x1, x2):
        # x = self.scattering(x)
        # x = x.view(-1, self.K, *self.scatt_output_shape)
        # x = self.bn(x)

        # Backbone

        x1 = self.resnet(x1)
        x2 = self.resnet(x2)
        '''
        x1 = self.custom_cnn(x1)
        x2 = self.custom_cnn(x2)
        '''
        return x1, x2

def load_weights(mdl, name):
    """Download pretrained state_dict and load into model.

    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        name {str} -- Name of dataset that was used to generate pretrained state_dict.

    Raises:
        ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
    """
    if name == 'vggface2':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
    elif name == 'casia-webface':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

    model_dir = os.path.join(get_torch_home(), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, os.path.basename(path))
    if not os.path.exists(cached_file):
        download_url_to_file(path, cached_file)

    state_dict = torch.load(cached_file)
    mdl.load_state_dict(state_dict)


def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home
