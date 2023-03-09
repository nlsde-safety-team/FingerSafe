from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import os
from FingerprintDataset import FingerprintTest, Fingerprint_Mask
from L_orientation import ridge_orient


def image_loader(loader, device, image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(unloader, tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)


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


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers,
                               style_layers, device):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        # print(layer)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# def run_style_transfer(cnn, normalization_mean, normalization_std,
#                        content_img, style_img, input_img, content_layers, style_layers, device,
#                        num_steps=300, style_weight=1000000, content_weight=1):
#     """Run the style transfer."""
#     print('Building the style transfer model..')
#     model, style_losses, content_losses = get_style_model_and_losses(cnn,
#                                                                      normalization_mean, normalization_std, style_img,
#                                                                      content_img, content_layers, style_layers, device)
#     attack_noise = torch.autograd.Variable(torch.zeros(2, 3, 224, 224).to(device))
#     attack_noise = attack_noise.detach()
#     attack_noise.requires_grad_(True)
#     input_img = input_img.detach()
#     output_img = input_img + attack_noise
#
#     optimizer = get_input_optimizer(attack_noise)
#     # optimizer = torch.optim.Adam([attack_noise], lr=0.0001)
#     optimizer.zero_grad()
#
#     print('Optimizing..')
#     run = [0]
#     while run[0] <= num_steps:
#         def closure():
#             # correct the values of updated input image
#             output_img = input_img + attack_noise
#             output_img.data.clamp_(0, 1)
#
#             optimizer.zero_grad()
#             model(output_img)
#             style_score = 0
#             content_score = 0
#
#             for sl in style_losses:
#                 style_score += sl.loss
#             for cl in content_losses:
#                 content_score += cl.loss
#
#             style_score *= style_weight
#             content_score *= content_weight
#
#             # loss = style_score + content_score
#             loss = content_score
#             loss.backward()
#
#             run[0] += 1
#             if run[0] % 50 == 0:
#                 print("run {}:".format(run))
#                 print('Style Loss : {:4f} Content Loss: {:4f}'.format(
#                     style_score.item(), content_score.item()))
#                 print()
#
#             return style_score + content_score
#
#         optimizer.step(closure)
#         attack_noise.data.clamp_(-8.0 / 255, 8.0 / 255)
#     # a last correction...
#     output_img = input_img + attack_noise
#     print(torch.max(attack_noise))
#     output_img.data.clamp_(0, 1)
#
#     return output_img

# def run_style_transfer(cnn, normalization_mean, normalization_std,
#                        content_img, style_img, input_img, content_layers, style_layers, device, conv_out,
#                        num_steps=300, style_weight=1000000, content_weight=1):
#     """Run the style transfer."""
#     print('Building the style transfer model..')
#     # model, style_losses, content_losses = get_style_model_and_losses(cnn,
#     #                                                                  normalization_mean, normalization_std, style_img,
#     #                                                                  content_img, content_layers, style_layers, device)
#     model = cnn
#     model(content_img)
#     content_emb = conv_out.features
#
#     attack_noise = torch.autograd.Variable(torch.zeros(2, 3, 224, 224).to(device))
#     attack_noise = attack_noise.detach()
#     attack_noise.requires_grad_(True)
#     input_img = input_img.detach()
#     output_img = input_img + attack_noise
#
#     optimizer = get_input_optimizer(attack_noise)
#     # optimizer = torch.optim.Adam([attack_noise], lr=0.0001)
#     optimizer.zero_grad()
#
#     print('Optimizing..')
#     run = [0]
#     while run[0] <= num_steps:
#         def closure():
#             # correct the values of updated input image
#             output_img = input_img + attack_noise
#             output_img.data.clamp_(0, 1)
#
#             optimizer.zero_grad()
#             model(output_img)
#             output_emb = conv_out.features
#             style_score = 0
#             content_score = 0
#
#             # for sl in style_losses:
#             #     style_score += sl.loss
#             # for cl in content_losses:
#             #     content_score += cl.loss
#             content_score = F.mse_loss(output_emb, content_emb)  # resnet
#
#             style_score *= style_weight
#             content_score *= content_weight
#
#             # loss = style_score + content_score
#             loss = content_score
#             loss.backward(retain_graph=True)
#
#             run[0] += 1
#             if run[0] % 50 == 0:
#                 print("run {}:".format(run))
#                 print('Content Loss: {:4f}, Orientation loss: {:4f}'.format(content_score.item(), loss_orient.item()))
#                 print()
#
#             return style_score + content_score
#
#         optimizer.step(closure)
#         attack_noise.data.clamp_(-8.0 / 255, 8.0 / 255)
#     # a last correction...
#     output_img = input_img + attack_noise
#     print(torch.max(attack_noise))
#     output_img.data.clamp_(0, 1)
#
#     return output_img

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, content_layers, style_layers, device, conv_out,
                       num_steps=300, style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    # model, style_losses, content_losses = get_style_model_and_losses(cnn,
    #                                                                  normalization_mean, normalization_std, style_img,
    #                                                                  content_img, content_layers, style_layers, device)
    model = cnn
    ridge = ridge_orient()

    content_ridge = ridge(content_img[0]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    for i in range(len(content_img)-1):
        tmp = ridge(content_img[i+1]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        content_ridge = torch.cat([content_ridge, tmp], dim=0)
    model(content_ridge)
    content_ridge_emb = conv_out.features

    attack_noise = torch.autograd.Variable(torch.zeros(2, 3, 224, 224).to(device))
    attack_noise = attack_noise.detach()
    attack_noise.requires_grad_(True)
    input_img = input_img.detach()
    output_img = input_img + attack_noise

    optimizer = get_input_optimizer(attack_noise)
    # optimizer = torch.optim.Adam([attack_noise], lr=0.0001)
    optimizer.zero_grad()

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            output_img = input_img + attack_noise
            output_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            output_ridge = ridge(output_img[0]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
            for i in range(len(output_img) - 1):
                tmp = ridge(output_img[i + 1]).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
                output_ridge = torch.cat([output_ridge, tmp], dim=0)
            model(output_ridge)
            output_ridge_emb = conv_out.features
            style_score = 0
            content_score = 0

            # for sl in style_losses:
            #     style_score += sl.loss
            # for cl in content_losses:
            #     content_score += cl.loss
            content_score = F.mse_loss(output_ridge_emb, content_ridge_emb)  # resnet

            style_score *= style_weight
            content_score *= content_weight
            loss_orient = L_orientation_target(ridge_orient(), content_img, output_img)

            # loss = style_score + content_score
            loss = loss_orient
            loss.backward(retain_graph=True)

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Content Loss: {:4f}, Orientation loss: {:4f}'.format(content_score.item(), loss_orient.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)
        attack_noise.data.clamp_(-8.0 / 255, 8.0 / 255)
    # a last correction...
    output_img = input_img + attack_noise
    print(torch.max(attack_noise))
    output_img.data.clamp_(0, 1)

    return output_img

# original version
# def run_style_transfer(cnn, normalization_mean, normalization_std,
#                        content_img, style_img, input_img, content_layers, style_layers, device,
#                        num_steps=300, style_weight=1000000, content_weight=1):
#     """Run the style transfer."""
#     print('Building the style transfer model..')
#     model, style_losses, content_losses = get_style_model_and_losses(cnn,
#                                                                      normalization_mean, normalization_std, style_img,
#                                                                      content_img, content_layers, style_layers, device)
#     optimizer = get_input_optimizer(input_img)
#
#     print('Optimizing..')
#     run = [0]
#     while run[0] <= num_steps:
#
#         def closure():
#             # correct the values of updated input image
#             input_img.data.clamp_(0, 1)
#
#             optimizer.zero_grad()
#             model(input_img)
#             style_score = 0
#             content_score = 0
#
#             for sl in style_losses:
#                 style_score += sl.loss
#             for cl in content_losses:
#                 content_score += cl.loss
#
#             style_score *= style_weight
#             content_score *= content_weight
#
#             # loss = style_score + content_score
#             loss = content_score
#             loss.backward()
#
#             run[0] += 1
#             if run[0] % 50 == 0:
#                 print("run {}:".format(run))
#                 print('Style Loss : {:4f} Content Loss: {:4f}'.format(
#                     style_score.item(), content_score.item()))
#                 print()
#
#             return style_score + content_score
#
#         optimizer.step(closure)
#
#     # a last correction...
#     input_img.data.clamp_(0, 1)
#
#     return input_img

# def run_style_transfer(cnn, normalization_mean, normalization_std,
#                        content_img, style_img, input_img, content_layers, style_layers, device,
#                        num_steps=300, style_weight=1, content_weight=1000):
#     """Run the style transfer.
#     here content_img is target image"""
#     print('Building the style transfer model..')
#     model, style_losses, content_losses = get_style_model_and_losses(cnn,
#                                                                      normalization_mean, normalization_std, style_img,
#                                                                      content_img, content_layers, style_layers, device)
#
#     # model = cnn
#     attack_noise = torch.autograd.Variable(torch.zeros(2, 3, 224, 224).to(device))
#     attack_noise = attack_noise.detach()
#     attack_noise.requires_grad_(True)
#     input_img = input_img.detach()
#     output_img = input_img + attack_noise
#
#     # model(content_img)
#     # content_emb = conv_out.features
#
#     optimizer = torch.optim.Adam([attack_noise], lr=0.0001)
#     optimizer.zero_grad()
#
#     print('Optimizing..')
#     run = [0]
#     while run[0] <= num_steps:
#         # correct the values of updated input image
#         # attack_noise = torch.clamp(attack_noise, -8.0/255, 8.0/255)
#         output_img = input_img + attack_noise
#         output_img.data.clamp_(0, 1)
#
#
#         model(output_img)
#         # output_emb = conv_out.features
#
#         # VGG
#         content_score = 0
#         for cl in content_losses:
#             content_score += cl.loss
#         # content_score = F.mse_loss(output_emb, content_emb)  # resnet
#
#         content_score = content_weight * content_score
#         loss_orient = L_orientation_target(ridge_orient(), content_img, output_img)
#
#         # loss = style_score + content_score
#         loss = content_score #+ 100*loss_orient
#         # loss.backward(retain_graph=True)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         # attack_noise.data.clamp_(-8.0/255, 8.0/255)
#
#         run[0] += 1
#         if run[0] % 50 == 0:
#             print("run {}:".format(run))
#             print('Content Loss: {:4f}, Orientation Loss: {:4f}'.format(content_score.item(), loss_orient.item()))
#             print()
#
#         # return style_score + content_score
#     # a last correction...
#     output_img.data.clamp_(0, 1)
#     print(torch.max(attack_noise))
#
#     return output_img


# def run_style_transfer(cnn, normalization_mean, normalization_std,
#                        content_img, style_img, input_img, content_layers, style_layers, device,
#                        num_steps=300, style_weight=1, content_weight=1000):
#     """Run the style transfer.
#     here content_img is target image"""
#     print('Building the style transfer model..')
#     model, style_losses, content_losses = get_style_model_and_losses(cnn,
#                                                                      normalization_mean, normalization_std, style_img,
#                                                                      content_img, content_layers, style_layers, device)
#
#     # model = cnn
#     attack_noise = torch.autograd.Variable(torch.zeros(2, 3, 224, 224).to(device))
#     attack_noise = attack_noise.detach()
#     attack_noise.requires_grad_(True)
#     input_img = input_img.detach()
#     output_img = input_img + attack_noise
#
#     # model(content_img)
#     # content_emb = conv_out.features
#
#     optimizer = torch.optim.Adam([attack_noise], lr=0.0001)
#     optimizer.zero_grad()
#
#     print('Optimizing..')
#     run = [0]
#     while run[0] <= num_steps:
#         # correct the values of updated input image
#         # attack_noise = torch.clamp(attack_noise, -8.0/255, 8.0/255)
#         output_img = input_img + attack_noise
#         output_img.data.clamp_(0, 1)
#
#
#         model(output_img)
#         # output_emb = conv_out.features
#
#         # VGG
#         content_score = 0
#         for cl in content_losses:
#             content_score += cl.loss
#         # content_score = F.mse_loss(output_emb, content_emb)  # resnet
#
#         content_score = content_weight * content_score
#         loss_orient = L_orientation_target(ridge_orient(), content_img, output_img)
#
#         # loss = style_score + content_score
#         loss = content_score #+ 100*loss_orient
#         # loss.backward(retain_graph=True)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         attack_noise.data.clamp_(-8.0/255, 8.0/255)
#
#         run[0] += 1
#         if run[0] % 50 == 0:
#             print("run {}:".format(run))
#             print('Content Loss: {:4f}, Orientation Loss: {:4f}'.format(content_score.item(), loss_orient.item()))
#             print()
#
#         # return style_score + content_score
#     # a last correction...
#     output_img.data.clamp_(0, 1)
#     print(torch.max(attack_noise))
#
#     return output_img


# def run_style_transfer_mask(cnn, normalization_mean, normalization_std,
#                        content_img, style_img, input_img, content_layers, style_layers, device, masks,
#                        num_steps=300, style_weight=1, content_weight=1000):
#     """Run the style transfer.
#     here content_img is target image"""
#     print('Building the style transfer model..')
#     model, style_losses, content_losses = get_style_model_and_losses(cnn,
#                                                                      normalization_mean, normalization_std, style_img,
#                                                                      content_img, content_layers, style_layers, device)
#
#     # model = cnn
#     attack_noise = torch.autograd.Variable(torch.zeros(2, 3, 224, 224).to(device))
#     attack_noise = attack_noise.detach()
#     attack_noise.requires_grad_(True)
#     input_img = input_img.detach()
#     output_img = input_img + attack_noise
#
#     # model(content_img)
#     # content_emb = conv_out.features
#
#     optimizer = torch.optim.Adam([attack_noise], lr=0.0001)
#     optimizer.zero_grad()
#
#     print('Optimizing..')
#     run = [0]
#     while run[0] <= num_steps:
#         # correct the values of updated input image
#         # attack_noise = torch.clamp(attack_noise, -8.0/255, 8.0/255)
#         output_img = input_img + attack_noise
#         output_img.data.clamp_(0, 1)
#
#
#         model(output_img*masks)
#         # output_emb = conv_out.features
#
#         # VGG
#         content_score = 0
#         for cl in content_losses:
#             content_score += cl.loss
#         # content_score = F.mse_loss(output_emb, content_emb)  # resnet
#
#         content_score = content_weight * content_score
#         loss_orient = L_orientation_target(ridge_orient(), content_img, output_img, masks)
#
#         # loss = style_score + content_score
#         loss = content_score + 0.1*loss_orient
#         # loss.backward(retain_graph=True)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         attack_noise.data.clamp_(-8.0/255, 8.0/255)
#
#         run[0] += 1
#         if run[0] % 50 == 0:
#             print("run {}:".format(run))
#             print('Content Loss: {:4f}, Orientation Loss: {:4f}'.format(content_score.item(), loss_orient.item()))
#             print()
#
#         # return style_score + content_score
#     # a last correction...
#     output_img.data.clamp_(0, 1)
#     print(torch.max(attack_noise))
#
#     return output_img

def L_orientation_target(ridge, t_images, adv_images, masks=None):
    # let orientation of adv_images close to target images
    distance = 0
    for i in range(len(t_images)):
        if masks is not None:
            mask = masks[i].bool()
            images_orient = ridge(t_images[i]).masked_fill(~mask.to('cuda'), value=0)
            adv_images_orient = ridge(adv_images[i]).masked_fill(~mask.to('cuda'), value=0)
        else:
            images_orient = ridge(t_images[i])
            adv_images_orient = ridge(adv_images[i])

        diff = torch.abs(adv_images_orient - images_orient)
        d = torch.sin(diff)
        d = d.abs()
        distance += torch.mean(d)
    return distance / (len(t_images))


def random_select(database, len):
    random_ind = random.randint(0, 1607)
    data = database.x_data[random_ind].cuda().unsqueeze(0)

    for i in range(len-1):
        random_ind = random.randint(0, 1607)
        tmp = database.x_data[random_ind].cuda().unsqueeze(0)
        data = torch.cat([data, tmp])
    return data


def export(paths, images):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    for idx in range(len(images)):
        p, f = os.path.split(paths[0][idx])
        if p.find('raw') != -1:
            new_path = p.replace('raw', 'perturb_content')
        else:
            new_path = p.replace('suibian', 'sample_content')
        os.makedirs(os.path.dirname(os.path.join(new_path, f)), exist_ok=True)
        img = unloader(images[idx].cpu().detach().squeeze(0)).convert('RGB')
        print("exporting to:" + str(os.path.join(new_path, f)))
        img.save(os.path.join(new_path, f))


class LayerActivations:
    features = None

    def __init__(self, model, layer):
        for name, m in model.named_modules():
            if name == layer:
                m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()


from torch.utils.data import Dataset, DataLoader
import random
from models.inception_resnet_v1 import ResNet
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # desired size of the output image
    imsize = 224
    batch_size = 2
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # model initialization
    model = ResNet(nclasses=268, classify=False)
    model.load_state_dict(torch.load('./best_models/clean_split_1009.pth'))
    # model = models.vgg11(pretrained=True).features
    cnn = model.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    conv_out = LayerActivations(model, 'features.layer1.2.conv2')

    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # load data
    test_set = FingerprintTest('./datasets/final/veri_sample_suibian')  # 68 classes
    # test_set = Fingerprint_Mask('./datasets/physical_square/evaluation', './datasets/physical_square/masks/evaluation')
    test_loader = DataLoader(dataset=test_set,
                              batch_size=batch_size,
                              shuffle=False)
    train_set = FingerprintTest('./datasets/final/train')  # 268 classes

    for batch_idx, (data, label, paths) in enumerate(test_loader):
        target_img = random_select(train_set, batch_size).to(device)  # return a batch of target image
        # input_img = content_img.clone()
        data = data.to(device)
        input_img = data.clone()
        # masks = masks.to(device)
        # if you want to use white noise instead uncomment the below line:
        # input_img = torch.randn(content_img.data.size(), device=device)
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    target_img, data, input_img, content_layers_default, style_layers_default, device, conv_out)
        export(paths, output)


    # loader = transforms.Compose([
    #     transforms.Resize(imsize),  # scale imported image
    #     transforms.ToTensor()])  # transform it into a torch tensor
    #
    # style_img = image_loader(loader, device, "./data/images/1-1.bmp")
    # content_img = image_loader(loader, device, "./data/images/6-1.bmp")

    # assert style_img.size() == content_img.size(), \
    #     "we need to import style and content images of the same size"

if __name__ == '__main__':
    main()