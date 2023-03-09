"""
  Purpose:  Train and save network weights
"""

import os
import torch
import torch.nn.functional as F
from FingerprintDataset import FingerprintTrain, FingerprintTest, FingerprintMix, Fingerprint_Seg
from torch.utils.data import Dataset, DataLoader
from models.inception_resnet_v1 import ResNet as Model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from kymatio import Scattering2D
import matplotlib.pyplot as plt
import numpy as np
from verification import get_valid_data, cal_embed, evaluate_lsm, evaluate
from identification import identify


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    model.classify = True
    nb_samples = 0
    for batch_idx, (data, target, _) in enumerate(train_loader):
        nb_samples += len(data)
        data, target = data.to(device), target.to(device)
        # preprocess_data = preprocess(data)
        optimizer.zero_grad()
        # output = model(data)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
            epoch,
            nb_samples,
            len(train_loader.dataset),
            100. * (batch_idx + 1) / len(train_loader),
            loss.item()), end='\r')


def train_all(type, device, dataroot, batch_size, epochs, valid_1, valid_2, issame_list, clean_1, clean_2, issame_clean
              ):
    acc_list_all, loss_list_all, epoch_list_all = [], [], []
    for i in range(1):
        train_set = FingerprintTrain(dataroot, 'train')
        # export(train_set.paths, train_set.x_data)
        # export(train_set.paths, train_set.masks)
        # np.save('./datasets/maps/15cm.npy', train_set.position)

        clean_dataroot = './datasets/physical_square/iden/iden_evaluation'
        adv_dataroot = './datasets/physical_square/iden/iden_evaluation'
        database_set = FingerprintTest(clean_dataroot, '2017_train')  # keep cleans
        query_set = FingerprintTest(adv_dataroot, '2017_test')  # training = clean

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True)

        model = Model(classify=True, nclasses=30).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        best_acc = 0.0
        acc_list, epoch_list = [], []
        model.train()
        for epoch in range(1, epochs + 1):
            print('Epoch: {}'.format(epoch))
            # scheduler.step(epoch)
            train(model, device, train_loader, optimizer, epoch)
            embedding1, embedding2 = cal_embed(valid_1, valid_2, model, embedding_size=1000, device=device)
            _, _, accuracy, _ = evaluate_lsm(41, embedding1, embedding2, issame_clean.to(device), 1000)
            print('verification acc = {}'.format(accuracy))
            iden_acc = identify(database_set, query_set, model, 1000)  # here is identify acc
            print('clean iden_acc = {}'.format(iden_acc))
            emb_clean1, emb_clean2 = cal_embed(clean_1, clean_2, model, embedding_size=1000, device=device)
            _, _, veri_acc, _ = evaluate_lsm(41, emb_clean1, emb_clean2, issame_clean.to(device), 1000)
            print('clean veri_acc = {}'.format(veri_acc))
            # acc_list.append(accuracy)
            # epoch_list.append(epoch)
            if accuracy > best_acc and epoch > 2:
                best_acc = accuracy
                file_name = 'physical_HG_{}_best.pth'.format(type)
                torch.save(model.state_dict(), os.path.join("saved_models", file_name))
                print("Saved: ", file_name)
            # if accuracy == 1:
            #     break
        # np.save('./result/acc_list_{}_B.npy'.format(type), acc_list)
        # np.save('./result/epoch_list_{}_B.npy'.format(type), epoch_list)
        acc_list_all.append(acc_list)
        epoch_list_all.append(epoch_list)
    return acc_list_all, loss_list_all, epoch_list_all


def main():
    """
    和 runfile.py 差不多，都是训练神经网络模型的
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    epochs = 30
    types = ['database']  # clean_split = train_split

    for type in types:
        print(type)
        adv_root = './datapaths/datapath_physical_HG_valid_{}_test.pkl'.format(type)
        valid_1, valid_2, issame_list, _, _ = get_valid_data(adv_root, flag=True, num=20)
        clean_1, clean_2, issame_clean, _, _ = get_valid_data('./datapaths/datapath_physical_HG_evaluation_test.pkl', num=20)
        dataroot = './datasets/physical_square/HG/{}_split'.format(type)
        acc_list, loss_list, epoch_list = train_all(type, device, dataroot, batch_size, epochs, valid_1, valid_2, issame_list,
                                                    clean_1, clean_2, issame_clean
                                                    )


if __name__ == '__main__':
    main()

