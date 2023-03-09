import torch, torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from models.inception_resnet_v1 import Siamese as Model
from FingerprintDataset import FingerprintTest, FingerprintTrain
import os
import cv2
from L_orientation import ridge_orient
import random

# https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py

import torch.nn.functional as F


# def L_FFT(images, adv_images):
#     distance = 0
#     for i in range(len(images)):
#         images_fft = get_fft_feature(images[i])
#         adv_images_fft = get_fft_feature(adv_images[i])
#         d = F.pairwise_distance(images_fft, adv_images_fft, p=2)
#         distance += torch.mean(d)
#     return distance / len(images)


def L_orientation(images, adv_images):
    # pairwise distance, 1 if similiar, 0 if dissimiliar
    distance = 0
    for i in range(len(images)):
        images_orient = ridge_orient(images[i])
        adv_images_orient = ridge_orient(adv_images[i])
        d = F.pairwise_distance(images_orient, adv_images_orient, p=2)
        # d = F.cosine_similarity(images_orient, adv_images_orient)
        distance += torch.mean(d)
    return distance / len(images)


def L_orientation_lsm(images, adv_images):
    # pairwise distance, 1 if similiar, 0 if dissimiliar
    distance = 0
    for i in range(len(images)):
        images_orient, reliability = ridge_orient(images[i])
        adv_images_orient, _ = ridge_orient(adv_images[i])
        diff = torch.abs(adv_images_orient - images_orient)
        # d = F.pairwise_distance(images_orient, adv_images_orient, p=2)
        d = torch.sin(diff)
        d = 1 / (d + 1e-2)
        d = torch.einsum('ij, ij -> ij', d, reliability)
        distance += torch.norm(d, p='fro')
    '''
    reliability = torch.pow(reliability, 5)
    reliability = reliability.cpu().detach().numpy()
    cv2.imshow('relia', reliability)
    cv2.waitKey(0)
    '''
    return distance / (224 * len(images))


def loss_content(attack_noise, data_pre):
    # hope to add on edge, larger the better.
    beta = 10  # according to dual-attention attack
    weighted_mask = beta * data_pre + 1
    weighted_noise = torch.einsum('ikl, ijkl -> ijkl', weighted_mask, attack_noise)
    return torch.norm(weighted_noise, p='fro')


def loss_smooth(attack_noise):
    s1 = torch.pow(attack_noise[:, :, :-1, :-1] - attack_noise[:, :, 1:, :-1], 2)
    s2 = torch.pow(attack_noise[:, :, :-1, :-1] - attack_noise[:, :, :-1, 1:], 2)
    return torch.mean(s1 + s2)


def random_select(database, len):
    random_ind = [random.randint(0, 1607) for _ in range(4)]  # choose 4 targets from train set
    data = [database.x_data[i].cuda() for i in random_ind]
    data_1, data_2, data_3, data_4 = data[0].unsqueeze(0), data[1].unsqueeze(0),\
                                     data[2].unsqueeze(0), data[3].unsqueeze(0)  # tensor
    for i in range(len-1):
        random_ind = [random.randint(0, 1607) for _ in range(4)]  # choose 4 targets from train set
        data = [database.x_data[i].cuda() for i in random_ind]
        data_1 = torch.cat([data_1, data[0].unsqueeze(0)])
        data_2 = torch.cat([data_2, data[1].unsqueeze(0)])
        data_3 = torch.cat([data_3, data[2].unsqueeze(0)])
        data_4 = torch.cat([data_4, data[3].unsqueeze(0)])
    return data_1, data_2, data_3, data_4


def test_final(embs, thresh=1.19):  # default label=1, negative
    from criterion import Criterion
    criterion = Criterion()
    avg_loss = 0
    batch = 8
    label = torch.tensor([1]).cuda()
    output1, output2 = embs[0], embs[1]
    True_Positive, True_Negative, False_Negative, False_Positive = 0, 0, 0, 0
    euc_dist = criterion(output1, output2, label)
    pred, tp, tn, fn, fp = criterion.calculate_metric_tensor_sigmoid(euc_dist, label, thresh)  # fixed thresh
    # print(label)
    # print(pred)

    True_Positive += tp
    True_Negative += tn
    False_Positive += fp
    False_Negative += fn

    margin_test = thresh
    # /2: half of the labels are not positive
    # tpr = True_Positive / (len(label) / (2))
    accuracy = (True_Positive + True_Negative) / (
            True_Positive + True_Negative + False_Positive + False_Negative)
    precision = True_Positive / (True_Positive + False_Positive)
    # print('veri_test loss:{:.4f}, threshold:{:.4f}, acc:{:.4f}'.format(float(avg_loss),
                                                                    #    float(margin_test), float(accuracy)))

    # True_Positive = True_Positive / (iterations * len(label) / 2)
    return pred


def train(model, device, train_loader, database, batch, iteration, draw=False):
    eps = 8./255.
    alpha = 0.2/255.
    TARGET_IMG_SIZE = 224
    attack_param = torch.zeros(train_loader.dataset.length, 3, TARGET_IMG_SIZE, TARGET_IMG_SIZE).to(device)
    model.eval()
    thresh = 1.19
    j = 0
    for batch_idx, (data, label, path) in enumerate(train_loader):
        # j += 1
        # if j <= 10:
        #     continue
        
        data, label = data.to(device), label.to(device)

        # randomly select from database
        data_1, data_2, data_3, data_4 = random_select(database, batch)  # batch*3*224*224
        data_1.requires_grad_(True)
        data_2.requires_grad_(True)
        data_3.requires_grad_(True)
        data_4.requires_grad_(True)

        attack_noise = torch.autograd.Variable(torch.zeros(batch, 3, TARGET_IMG_SIZE, TARGET_IMG_SIZE).to(device))
        # attack_noise = attack_noise * epsilon
        # attack_noise = attack_noise.clamp(-epsilon, epsilon)
        attack_noise = attack_noise.detach()
        data_adv = attack_noise + data
        data_adv = torch.clamp(data_adv, min=0, max=1).detach()
        # attack_noise.requires_grad_(True)
        # data = data.detach()
        # optimizer = torch.optim.Adam([attack_noise], lr=0.0001)
        # optimizer.zero_grad()

        # data_pre = torch.where(data_pre > 0.1, 1, 0)
        losses = []
        for i in range(iteration):
            # attack_noise = torch.einsum('ijkl, ikl -> ijkl', attack_noise, data_pre)
            # attack_noise = attack_noise.detach()
            # attack_noise.requires_grad = True
            # data = data.detach()
            # optimizer = torch.optim.Adam([attack_noise], lr=0.0001)
            # optimizer.zero_grad()
            # attack_noise.requires_grad_(True)
            # data_adv = data + attack_noise
            data_adv.requires_grad_(True)
            '''
            output = model(data)
            print('target = {}'.format(target))
            print('predict = {}'.format(output.max(1, keepdim=True)[1]))
            '''
            emb_1, emb_2, emb_3, emb_4 = model(data_adv, data_1), model(data_adv, data_2), \
                                                 model(data_adv, data_3), model(data_adv, data_4)  # output rep, not score
            score_1, score_2, score_3, score_4 = test_final(emb_1), test_final(emb_2), \
                                                 test_final(emb_3), test_final(emb_4),
            zero_tensor = torch.zeros((batch, 1)).cuda()
            loss_rep = torch.max(zero_tensor, score_1 - 1 / 2) + torch.max(zero_tensor, score_2 - 1 / 2) + \
                       torch.max(zero_tensor, score_3 - 1 / 2) + torch.max(zero_tensor, score_4 - 1 / 2)
            loss_rep = torch.sum(loss_rep)
            loss_dist = torch.sum(F.pairwise_distance(data_adv, data))

            # loss = 1e-1 * loss_class - (3e-2 * loss_content(attack_noise, data_pre)) + 1e4 * loss_smooth(attack_noise)
            # loss = 1 * loss_rep + 1 * loss_dist
            loss = loss_rep 
            losses.append(loss.detach().cpu())
            # loss = 1e-1 * loss_class - (3e-2 * loss_content(attack_noise, data_pre))
            grad = torch.autograd.grad(loss, data_adv,
                                       retain_graph=False, create_graph=False)[0]

            data_adv = data_adv.detach() - alpha * grad.sign()
            delta = torch.clamp(data_adv - data, min=-eps, max=eps)
            data_adv = torch.clamp(data + delta, min=0, max=1).detach()
            delta = data_adv - data
            print(delta.sum())
            # loss.backward()
            # optimizer.step()
            # attack_noise = attack_noise.clamp(-epsilon, epsilon)
        if draw:
            losses = np.array(losses)
            print(losses)
            np.save('./convergence/workshop.npy', losses)
            exit(0)
        print(torch.max(data_adv - data))
        print('noise added')

        attack_noise = (data_adv[0, ...] - data[0, ...]) + (8 / 255)
        attack_noise = attack_noise.permute(1, 2, 0).abs().detach().cpu().numpy() * 255 * 255 / 8
        attack_noise = attack_noise.astype('uint8')
        cv2.imshow('attack_noise', attack_noise)

        data_adv = (data_adv[0, ...]).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255
        data_adv = data_adv.astype('uint8')
        cv2.imwrite('noise.bmp', data_adv)
        cv2.imshow('data', data_adv)

        cv2.waitKey(0)
        # print(attack_noise)
        exit(0)
        attack_param[index] = attack_noise
        '''
        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
            epoch,
            nb_samples,
            len(train_loader.dataset),
            100. * (batch_idx + 1) / len(train_loader),
            loss.item()), end='\r')
        '''


def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # preprocess_data = preprocess(data)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


def export(paths, images):
    unloader = torchvision.transforms.ToPILImage()
    rgb2bgr = [2, 1, 0]
    images = images[:, rgb2bgr, :, :]
    for idx in range(len(images)):
        p, f = os.path.split(paths[0][idx])
        if p.find('train') != -1:
            new_path = p.replace('train', 'perturb')
        else:
            new_path = p.replace('test', 'sample')
        os.makedirs(os.path.dirname(os.path.join(new_path, f)), exist_ok=True)
        img = unloader(images[idx].cpu().detach().squeeze(0)).convert('RGB')
        print("exporting to:" + str(os.path.join(new_path, f)))
        img.save(os.path.join(new_path, f))


def run_adv(epoch=-1):
    print('adversarial attacking: Optimization based')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_name = 'colored_sigmoid_0.8780.pth'
    train_root = 'datasets/final/train'
    test_root = 'datasets/final/veri_test'
    iteration = 20
    TARGET_IMG_SIZE = 224
    batch_size = 1
    model = Model(nclasses=268, device=device).to(device)
    # model.load_state_dict(torch.load(os.path.join("saved_models", weights_name)))
    model.load_state_dict()
    model.eval()
    dataset_train = FingerprintTrain(train_root)
    dataset_test = FingerprintTest(test_root)
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(dataset=dataset_test,
                              batch_size=batch_size,
                              shuffle=False)
    for param in model.parameters():
        param.requires_grad = False
    # attacker = PGD(model)
    train(model, device, train_loader, dataset_train, batch_size, iteration, draw=True)
    test_loss, _ = validate(model, device, test_loader)


if __name__ == '__main__':
    run_adv()
