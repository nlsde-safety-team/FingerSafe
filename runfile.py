"""
  Purpose:  Train and save network weights
"""

import os
import torch
import torch.nn.functional as F
from FingerprintDataset import FingerprintTrain, FingerprintTest
from torch.utils.data import Dataset, DataLoader
from models.inception_resnet_v1 import ResNet as Model
import numpy as np
from verification import get_valid_data, cal_embed, evaluate_lsm
from identification import identify


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    model.classify = True
    nb_samples = 0
    for batch_idx, (data, target, _) in enumerate(train_loader):
        nb_samples += len(data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
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


def train_all(type, device, dataroot, batch_size, epochs, valid_1, valid_2, issame_list,
              clean_1, clean_2, issame_clean, database_set, query_set
              ):
    np.random.seed(0)
    train_set = FingerprintTrain(dataroot, 'train')

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True)

    model = Model(classify=True, nclasses=268).to(device)  # 训练神经网络，依据facenet代码，classify=True，
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_acc = 0.0
    model.train()
    embedding_size = 1000
    for epoch in range(1, epochs + 1):
        print('Epoch: {}'.format(epoch))
        # scheduler.step(epoch)
        train(model, device, train_loader, optimizer, epoch)
        embedding1, embedding2 = cal_embed(valid_1, valid_2, model, embedding_size=embedding_size, device=device)
        eval_loss, tpr, accuracy, thrsh = evaluate_lsm(201, embedding1, embedding2, issame_list.to(device), embedding_size)
        print('verification acc = {}'.format(accuracy))

        # 用当前模型测一下在干净样本下的verification和identification--->training stage任务
        iden_acc = identify(database_set, query_set, model, embedding_size)  # here is identify acc
        print('clean iden_acc = {}'.format(iden_acc))
        emb_clean1, emb_clean2 = cal_embed(clean_1, clean_2, model, embedding_size=embedding_size, device=device)
        _, veri_tpr, veri_acc, _ = evaluate_lsm(201, emb_clean1, emb_clean2, issame_clean.to(device), embedding_size)
        print('clean veri_acc = {}, clean_veri_tpr={}'.format(veri_acc, veri_tpr))

        # 根据verification acc来决定保存最优模型
        if accuracy >= best_acc and epoch > 2:
            best_acc = accuracy
            file_name = 'workshop_{}_best.pth'.format(type)
            torch.save(model.state_dict(), os.path.join("saved_models", file_name))
            print("Saved: ", file_name)



def main():
    """
    train a DNN for fingerprint classification
    train_root: training sample (268 class in HKPolyU), splitted as train/test.
    test_root: testing sample (.pkl file of test set in 268 class), used to evaluate the verification result in sample.
    clean_dataroot, adv_dataroot, database_set, query_set: test the identification accuracy of this model. no need to change this.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    epochs = 30
    types = ['strong', 'weak']  # clean_split = train_split

    for type in types:
        print(type)
        dir = './datasets/final/'
        train_root = 'perturb_workshop_{}_split'.format(type)
        dataroot = os.path.join(dir, train_root)
        test_root = './datapaths/datapath_valid_workshop_{}'.format(type) + '_test.pkl'
        # sample image pair from test_root
        valid_1, valid_2, issame_list, _, _ = get_valid_data(test_root, flag=True)
        # sample clean pair from dataroot
        clean_1, clean2, issame_clean, _, _ = get_valid_data('./datapaths/datapath_clean_test.pkl')
        clean_dataroot = './datasets/final_identification/iden_clean'
        adv_dataroot = './datasets/final_identification/iden_clean'
        database_set = FingerprintTest(clean_dataroot, 'training')  # keep clean
        query_set = FingerprintTest(adv_dataroot, 'testing')  # training = clean
        train_all(type, device, dataroot, batch_size, epochs,
                                                    valid_1, valid_2, issame_list, clean_1, clean2, issame_clean,
                                                    database_set, query_set)

if __name__ == '__main__':
    main()

