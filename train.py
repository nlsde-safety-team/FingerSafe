# coding:utf-8
"""
  Purpose:  Train and save network weights
"""

import os
import torch
import torch.nn.functional as F
from FingerprintDataset import FingerprintVeri
from torch.utils.data import Dataset, DataLoader
from models.inception_resnet_v1 import Siamese as Model
from kymatio import Scattering2D
from criterion import Criterion
import numpy as np
import logging
from logger_v1 import setup_logs


def train(model, device, train_loader, optimizer, epoch, criterion, logger):
    model.train()
    nb_samples = 0
    loss_all = []
    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        nb_samples += len(data1)
        data1, data2, target = data1.float().to(device), data2.float().to(device), target.to(device)
        optimizer.zero_grad()
        output1, output2 = model(data1, data2)
        _, loss = criterion(output1, output2, target) 
        loss.backward()
        optimizer.step()
        loss_all.append(loss.item())
        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f},'.format(
                epoch,
                nb_samples,
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                loss.item()))
            # print(euc_dist)
            # print(target)
    logger.info('Train Epoch: {}, Loss: {:.6f},'.format(
        epoch,
        np.mean(loss_all)))
    return np.mean(loss_all)


def validate(model, device, test_loader, criterion, logger):
    model.eval()
    test_loss = 0
    correct = 0
    loss_all = []
    True_Positive, True_Negative, False_Negative, False_Positive = torch.zeros(1000), torch.zeros(1000), torch.zeros(1000), torch.zeros(1000)
    with torch.no_grad():
        for i, (data1, data2, target) in enumerate(test_loader):
            data1, data2, target = data1.float().to(device), data2.float().to(device), target.to(device)
            output1, output2 = model(data1, data2)
            euc_dist, loss = criterion(output1, output2, target)
            threshold, tp, tn, fn, fp = criterion.calculate_metric_tensor_thresh_sigmoid(euc_dist, target)

            True_Positive += tp
            True_Negative += tn
            False_Positive += fp
            False_Negative += fn

            loss_all.append(loss.item())  # sum up batch loss
    precision = True_Positive / (True_Positive + False_Positive)
    recall = True_Positive / (True_Positive + False_Negative)
    accuracy = (True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative)
    F1_score = 2 * True_Positive / (2 * True_Positive + False_Negative + False_Positive)

    EER_temp = -torch.abs(False_Positive - False_Negative)
    margin_index = torch.argmax(EER_temp)
    F1_score_max = F1_score[margin_index]
    precision_max = precision[margin_index]
    recall_max = recall[margin_index]
    accuracy_max = accuracy[margin_index]
    threshold_final = threshold[margin_index]
    tpr = True_Positive[margin_index] / (len(test_loader.dataset) / 2)

    test_loss = np.mean(loss_all)
    logger.info('\nTest set: Average loss: {:.4f}, in threshold {:.2f}, Precision: {:.4f}, Recall: {:.4f}, Acc: {:.4f}, TPR: {:.4f})'.format(
        test_loss,
        threshold_final,
        precision_max,
        recall_max,
        accuracy_max,
        tpr))

    return test_loss, (False_Positive[margin_index]+False_Negative[margin_index])/2, tpr, accuracy_max, threshold_final



def validate_sigmoid(model, device, test_loader, criterion, logger):
    model.eval()
    test_loss = 0
    correct = 0
    loss_all = []
    True_Positive, True_Negative, False_Negative, False_Positive = torch.zeros(1), torch.zeros(1),\
                                                                   torch.zeros(1), torch.zeros(1),
    with torch.no_grad():
        for i, (data1, data2, target) in enumerate(test_loader):
            data1, data2, target = data1.float().to(device), data2.float().to(device), target.to(device)
            # preprocess_data = preprocess(data)
            output1, output2 = model(data1, data2)
            euc_dist, loss = criterion(output1, output2, target)
            score, tp, tn, fn, fp = criterion.calculate_metric_tensor_sigmoid(euc_dist, target)

            True_Positive += tp
            True_Negative += tn
            False_Positive += fp
            False_Negative += fn

            loss_all.append(loss.item())  # sum up batch loss
    #precision = True_Positive / (True_Positive + False_Positive)
    #recall = True_Positive / (True_Positive + False_Negative)
    accuracy = (True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative)
    F1_score = 2 * True_Positive / (2 * True_Positive + False_Negative + False_Positive)

    EER_temp = -torch.abs(False_Positive - False_Negative)
    margin_index = torch.argmax(accuracy)
    # F1_score_max = F1_score[margin_index]
    # precision_max = precision[margin_index]
    # recall_max = recall[margin_index]
    # accuracy_max = accuracy[margin_index]
    # threshold_final = threshold[margin_index]
    tpr = True_Positive / (len(test_loader.dataset) / 2)

    test_loss = np.mean(loss_all)
    print(test_loss)
    print(accuracy.item())
    print(tpr.item())
    # print('\nTest set: Average loss: {:.4f}, Acc: {}, TPR: {})'.format(
    #     test_loss,
    #     accuracy[0],
    #     tpr[0]))

    # test_sample_num = 68 * 6
    # True_Positive, True_Negative, False_Positive, False_Negative = True_Positive/test_sample_num, True_Negative/test_sample_num, False_Positive/test_sample_num, False_Negative/test_sample_num
    # logger.info('TP: {:.4f}, TN {:.4f}, FP: {:.4f}, FN: {:.4f}'.format(
    #     True_Positive[margin_index], True_Negative[margin_index], False_Positive[margin_index], False_Negative[margin_index]
    # ))
    return test_loss.item(), (False_Positive[margin_index]+False_Negative[margin_index])/2, tpr.item(), accuracy.item()


def main():
    device = torch.device('cuda:0')
    train_pkl = './datapaths/datapath_MHS_clean_train.pkl'
    test_pkl = './datapaths/datapath_MHS_clean_test.pkl'
    batch_size = 1
    epochs = 3

    logging_dir = './saved_models'
    run_name = 'normal_validation'

    logger = setup_logs(logging_dir, run_name)
    logger.info("batch_size:{}".format(batch_size))

    train_set = FingerprintVeri(train_pkl, num=268)
    test_set = FingerprintVeri(test_pkl, num=68)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False)

    model = Model(nclasses=268, device=device).to(device)
    model.load_state_dict(torch.load('./saved_models/MHS_th_sigmoid_0.7744.pth'))
    criterion = Criterion(margin=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_all = []
    test_all = []
    best_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, criterion, logger)
        test_loss, EER, tpr, acc, thresh = validate(model, device, test_loader, criterion, logger)
        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            file_name = "MHS_th_{}_sigmoid_{:.4f}.pth".format(thresh, best_acc)
            torch.save(model.state_dict(), os.path.join("saved_models", file_name))
            print("Saved: ", file_name)
        train_all.append(1-EER)
        test_all.append(acc)

    test_loss, EER, tpr, _ = validate_sigmoid(model, device, test_loader, criterion, logger)
    print(test_loss, EER, tpr)


if __name__ == '__main__':
    main()
