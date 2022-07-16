import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
class Criterion(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2):
        super(Criterion, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        # N x 1 distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        return euclidean_distance

    # 用于evaluate
    def calculate_metric_tensor(self, euclidean_distance, label, size=1000):
        threshold = torch.linspace(0, self.margin * 2, steps=size).cuda()   # 1000
        predicted_label = torch.where(euclidean_distance > threshold, torch.tensor(1).cuda(), torch.tensor(0).cuda())  # 8*1000

        '''
                               Predicted
                         True     False
        Actual  True      TP       FN
                 False    FP       TN
        '''
        label = label.squeeze().repeat(size, 1).transpose(0, 1)
        True_Negative = torch.sum(torch.logical_and(predicted_label, label), dim=0)
        True_Positive = torch.sum(torch.logical_not(torch.logical_or(predicted_label, label)), dim=0)
        False_Positive = torch.sum(torch.logical_and(torch.logical_not(predicted_label), label), dim=0)
        False_Negative = torch.sum(torch.logical_and(predicted_label, torch.logical_not(label)), dim=0)

        return threshold, True_Positive.detach().cpu(), True_Negative.detach().cpu(), False_Negative.detach().cpu(), False_Positive.detach().cpu()

    # Used for Malhotra, et. al
    def calculate_metric_tensor_thresh_sigmoid(self, euclidean_distance, label, size=1000):
        threshold = torch.linspace(0, self.margin * 2, steps=size).cuda()   # 1000
        euclidean_distance = euclidean_distance.squeeze().repeat(1, size)  # batch * 1000
        predicted_score = torch.sigmoid(euclidean_distance - threshold)
        predicted_label = torch.where(predicted_score < 1/2, torch.tensor(0).cuda(), torch.tensor(1).cuda())

        # print(predicted_label.shape)

        '''
                               Predicted
                         True     False
        Actual  True      TP       FN
                 False    FP       TN
        '''
        label = label.squeeze().repeat(size, 1).transpose(0, 1)
        True_Negative = torch.sum(torch.logical_and(predicted_label, label), dim=0)
        True_Positive = torch.sum(torch.logical_not(torch.logical_or(predicted_label, label)), dim=0)
        False_Positive = torch.sum(torch.logical_and(torch.logical_not(predicted_label), label), dim=0)
        False_Negative = torch.sum(torch.logical_and(predicted_label, torch.logical_not(label)), dim=0)

        return threshold, True_Positive.detach().cpu(), True_Negative.detach().cpu(), False_Negative.detach().cpu(), False_Positive.detach().cpu()

    # Used for Malhotra et, al.
    def calculate_metric_tensor_sigmoid(self, euclidean_distance, label, thresh=1.19):
        predicted_score = torch.sigmoid(euclidean_distance - thresh)
        predicted_label = torch.where(predicted_score < 1/2, torch.tensor(0).cuda(), torch.tensor(1).cuda())
        # print(predicted_label.shape)

        '''
                               Predicted
                         True     False
        Actual  True      TP       FN
                 False    FP       TN
        '''
        # label = label.squeeze().repeat(size, 1).transpose(0, 1)
        True_Negative = torch.sum(torch.logical_and(predicted_label, label), dim=0)
        True_Positive = torch.sum(torch.logical_not(torch.logical_or(predicted_label, label)), dim=0)
        False_Positive = torch.sum(torch.logical_and(torch.logical_not(predicted_label), label), dim=0)
        False_Negative = torch.sum(torch.logical_and(predicted_label, torch.logical_not(label)), dim=0)

        return predicted_score, True_Positive.detach().cpu(), True_Negative.detach().cpu(), False_Negative.detach().cpu(), False_Positive.detach().cpu()


    def calculate_metric(self, euclidean_distance, label, size=1000):
        # threshold = np.linspace(0, self.margin * 2, steps=1000)  # 1000
        threshold = np.arange(0, self.margin * 2, 0.004)  # 1000
        predicted_label = np.where(euclidean_distance > threshold, 1, 0)  # 8*1000

        '''
                               Predicted
                         True     False
        Actual  True      TP       FN
                 False    FP       TN
        '''
        label = label.squeeze().repeat(size, 1).transpose(0, 1)
        label = label.cpu().numpy()
        True_Positive = np.sum(np.logical_not(np.logical_or(predicted_label, label)), dim=0)
        False_Positive = np.sum(np.logical_and(np.logical_not(predicted_label), label), dim=0)
        True_Negative = np.sum(np.logical_and(predicted_label, label), dim=0)
        False_Negative = np.sum(np.logical_and(predicted_label, np.logical_not(label)), dim=0)

        return threshold, True_Positive, True_Negative, False_Negative, False_Positive

    def calculate_accuracy(self, threshold, dist, actual_issame):
        actual_issame = actual_issame[:, 0]
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
        is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / dist.size
        return tpr, fpr, acc, is_fp, is_fn


    def calculate_TPR(self, euclidean_distance, threshold, label):
        predicted_label = torch.where(euclidean_distance > threshold, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        predicted_label = predicted_label.cuda()
        label = label.cuda()

        True_Negative = torch.sum(torch.logical_and(predicted_label, label), dim=0)
        True_Positive = torch.sum(torch.logical_not(torch.logical_or(predicted_label, label)), dim=0)
        False_Positive = torch.sum(torch.logical_and(torch.logical_not(predicted_label), label), dim=0)
        False_Negative = torch.sum(torch.logical_and(predicted_label, torch.logical_not(label)), dim=0)
        return True_Positive.detach().cpu(), True_Negative.detach().cpu(), False_Negative.detach().cpu(), \
               False_Positive.detach().cpu()









