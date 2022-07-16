# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os
from models.inception_resnet_v1 import ResNet as ResNet
from data import DataSampler,  DataSampler_adv
import matplotlib.pyplot as plt

#### Evaluate embeddings by using distance metrics to perform verification on the official LFW veri_test set.
"""
The functions in the next block are copy pasted from `facenet.src.lfw`. Unfortunately that module has an absolute import from `facenet`, so can't be imported from the submodule

added functionality to return false positive and false negatives
"""
#%%

from sklearn.model_selection import KFold
from scipy import interpolate

# LFW functions taken from David Sandberg's FaceNet implementation
import math
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive = []
    is_false_negative = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _ ,_ = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)  # here acc_train最大也就0.5+，肯定有问题？
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)

    return tpr, fpr, accuracy, is_false_positive, is_false_negative, best_threshold_index


def evaluate_lsm(iterations, embeddings1, embeddings2, issame, embedding_size=1000):
    from criterion import Criterion
    criterion = Criterion()
    avg_loss = 0
    True_Positive, True_Negative, False_Negative, False_Positive = torch.zeros(embedding_size), torch.zeros(
        embedding_size), torch.zeros(embedding_size), torch.zeros(embedding_size)
    batch = 8
    with torch.no_grad():
        for e in range(iterations):
            # siamese_1, siamese_2, label = veri_sampler.sample()
            # siamese_1, siamese_2, label = siamese_1.cuda(), siamese_2.cuda(), label.cuda()

            # output1 = model(siamese_1)
            # output2 = model(siamese_2)
            output1 = embeddings1[e * batch: (e + 1) * batch]
            output2 = embeddings2[e * batch: (e + 1) * batch]
            label = issame[e * batch: (e + 1) * batch]

            euc_dist = criterion(output1, output2, label)

            # avg_loss = avg_loss + float(loss.item())

            threshold, tp, tn, fn, fp = criterion.calculate_metric_tensor(euc_dist, label, size=embedding_size)

            True_Positive += tp
            True_Negative += tn
            False_Positive += fp
            False_Negative += fn


    precision = True_Positive / (True_Positive + False_Positive)
    recall = True_Positive / (True_Positive + False_Negative)
    accuracy = (True_Positive + True_Negative) / (
            True_Positive + True_Negative + False_Positive + False_Negative)
    F1_score = 2 * True_Positive / (2 * True_Positive + False_Negative + False_Positive)

    # EER_temp = -torch.abs(False_Positive - False_Negative)
    # margin_index = torch.argmax(EER_temp)  # todo: use EER to determine thresh
    margin_index = torch.argmax(accuracy)  # todo: here use accuracy to determine best thresh
    margin_train = threshold[margin_index]
    F1_score_max = F1_score[margin_index]
    precision_max = precision[margin_index]
    recall_max = recall[margin_index]
    accuracy_max = accuracy[margin_index]
    threshold_final = threshold[margin_index]
    # /2: half of the labels are not positive
    tpr = True_Positive[margin_index] / (iterations * len(label) / 2)
    fpr = False_Positive / (False_Positive + True_Negative)
    gen_plot(fpr, recall)
    # print(True_Positive[margin_index], True_Negative[margin_index], False_Positive[margin_index], False_Negative[margin_index])
    print(avg_loss / iterations, tpr, F1_score_max, precision_max, recall_max, accuracy_max, threshold_final)
    # print("acc={}, tpr={}, fpr={}, precision={}, F1_score={}, recall={}"
    #       .format(accuracy_max, tpr, fpr, precision_max, F1_score_max, recall_max))
    return avg_loss / iterations, tpr, accuracy_max, threshold_final


def test_final(iterations, embeddings1, embeddings2, issame,  thres):
    from criterion import Criterion
    criterion = Criterion()
    avg_loss = 0
    batch = 8
    True_Positive, True_Negative, False_Negative, False_Positive = 0, 0, 0, 0
    with torch.no_grad():
        for e in range(iterations):
            output1 = embeddings1[e * batch: (e + 1) * batch]
            output2 = embeddings2[e * batch: (e + 1) * batch]
            label = issame[e * batch: (e + 1) * batch]

            euc_dist = criterion(output1, output2, label)
            tp, tn, fn, fp = criterion.calculate_TPR(euc_dist, thres, label)  # fixed thresh

            True_Positive += tp
            True_Negative += tn
            False_Positive += fp
            False_Negative += fn

        margin_test = thres
        # /2: half of the labels are not positive
        tpr = True_Positive / (iterations * len(label) / (2))
        accuracy = (True_Positive + True_Negative) / (
                True_Positive + True_Negative + False_Positive + False_Negative)
        precision = True_Positive / (True_Positive + False_Positive)
        print('veri_test loss:{:.4f}, threshold:{:.4f}, tpr:{:.4f}, acc:{:.4f}'.format(float(avg_loss / iterations),
                                                                       float(margin_test), float(tpr), float(accuracy)))

    # True_Positive = True_Positive / (iterations * len(label) / 2)
    return avg_loss / iterations, tpr


def calculate_accuracy(threshold, dist, actual_issame):
    actual_issame = actual_issame[:, 0]
    predict_issame = np.greater(dist, threshold)  # less to greater
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # tp = np.sum(np.logical_not(np.logical_or(predict_issame, actual_issame)))
    # fp = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    # tn = np.sum(np.logical_and(predict_issame, actual_issame))
    # fn = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc, is_fp, is_fn

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings1, embeddings2, actual_issame, nrof_folds=4, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    # embeddings1 = embeddings[0::2]
    # embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn, best_index = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame.cpu()), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    print(accuracy, tpr)
    gen_plot(fpr, tpr)
    # thresholds = np.arange(0, 4, 0.001)
    # val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
    #     np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr[best_index], fpr, np.mean(accuracy)  #, val, val_std, far, fp, fn


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve-perturb_fingeradv0912B", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    plt.savefig('ROC_fingeradv_test.jpg')
    plt.close()


def get_valid_data(pkl_path, pkl_path_adv=None, flag=False, num=68):
    print(pkl_path)
    print(pkl_path_adv)
    add1, add2 = [], []
    if pkl_path_adv:
        data_sampler = DataSampler_adv(num, 4, pkl_path, pkl_path_adv, mode='rgb')
    else:
        data_sampler = DataSampler(num, 4, pkl_path, mode='rgb')  # here should be 68 classes

    np.random.seed(0)
    siamese_1, siamese_2, label, ad1, ad2 = data_sampler.sample(flag=flag)

    valid_1 = siamese_1
    valid_2 = siamese_2
    valid_issame = label
    add1.append(ad1)
    add2.append(ad2)
    for i_pair in range(200):
        siamese_1, siamese_2, label, ad1, ad2 = data_sampler.sample(flag)
        valid_1 = torch.cat([valid_1, siamese_1], dim=0)
        valid_2 = torch.cat([valid_2, siamese_2], dim=0)
        valid_issame = torch.cat([valid_issame, label], dim=0)  # pairs
        add1.append(ad1)
        add2.append(ad2)
    return valid_1, valid_2, valid_issame, add1, add2


def cal_embed(valid_1, valid_2, model, embedding_size, device):
    batch_size = 4
    idx = 0
    embedding_1 = np.zeros([len(valid_1), embedding_size])
    embedding_2 = np.zeros([len(valid_2), embedding_size])
    model.eval()
    model.classify = False  # embedding计算的是表示空间，不用分类
    with torch.no_grad():
        while idx + batch_size <= len(embedding_1):
            batch = valid_1[idx:idx + batch_size].clone().detach()
            embedding_1[idx:idx + batch_size] = model(batch.to(device)).cpu()
            batch = valid_2[idx:idx + batch_size].clone().detach()
            embedding_2[idx:idx + batch_size] = model(batch.to(device)).cpu()
            idx += batch_size
    embedding_1 = torch.from_numpy(embedding_1).to(device)
    embedding_2 = torch.from_numpy(embedding_2).to(device)
    return embedding_1, embedding_2


def validate(model, device, test_loader):
    model.eval()
    model.classify = True
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred = output.data.max(1)[1]
            print('target = {}'.format(target))
            print('predict = {}'.format(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


def Scat(valid_1, valid_2, size=1*81*6*6):
    from kymatio import Scattering2D
    scattering = Scattering2D(J=2, shape=(50, 50))
    if torch.cuda.is_available():
        print("Move scattering to GPU")
        scattering = scattering.cuda()
        device = 'cuda'

    batch_size = 4
    idx = 0
    embedding_size = size
    embedding_1 = np.zeros([len(valid_1), embedding_size])
    embedding_2 = np.zeros([len(valid_2), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(embedding_1):
            batch = valid_1[idx:idx + batch_size].clone().detach()
            emb = scattering(batch.cuda()).cpu()
            embedding_1[idx:idx + batch_size] = emb.view(emb.size(0), -1)
            batch = valid_2[idx:idx + batch_size].clone().detach()
            emb = scattering(batch.cuda()).cpu()
            embedding_2[idx:idx + batch_size] = emb.view(emb.size(0), -1)
            idx += batch_size
    embedding_1 = torch.from_numpy(embedding_1).to(device)
    embedding_2 = torch.from_numpy(embedding_2).to(device)
    return embedding_1, embedding_2


if __name__ == '__main__':
    """
    used in verification, calculate verification TPR
    change following parameters when you use it:
    - weight_name: directory of trained model
    - root: save the .pkl file in clean samples
    - adv_type: type of protection
    - root_adv: the saved .pkl file for all paths for adv sample directory
    - embedding_size: embedding shape in different backbone. normally we use resnet=1000
    - can use scatnet for feature extraction
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    resnet = ResNet(
        classify=False,
        nclasses=268
    ).to(device)
    weight_name = './best_models/MHS_clean_best_1030.pth'
    print(weight_name)
    resnet.load_state_dict(torch.load(weight_name))
    resnet.eval()

    root = './datapaths/datapath_MHS_clean_test.pkl'  # clean
    clean_1, clean_2, issame_clean, _, _ = get_valid_data(root, num=68)

    adv_type = ['fingersafe']
    for a in adv_type:
        root_adv = './datapaths/datapath_MHS_{}_test.pkl'.format(a)  # adv
        valid_1, valid_2, issame_list, _, _ = get_valid_data(root, root_adv, num=68)

        # embedding size: resnet=1000, densenet=1024, inceptionV3=2048
        embedding1, embedding2 = cal_embed(valid_1, valid_2, resnet, embedding_size=1000, device=device)
        emb_clean1, emb_clean2 = cal_embed(clean_1, clean_2, resnet, embedding_size=1000, device=device)

        # used for scatnet. remember the input shape should be 50*50
        # embedding1, embedding2 = Scat(valid_1, valid_2, size=11664)
        # emb_clean1, emb_clean2 = Scat(clean_1, clean_2, size=11664)

        eval_loss, tpr, accuracy, thrsh = evaluate_lsm(201, emb_clean1, emb_clean2, issame_clean.to(device))
        test_final(201, embedding1, embedding2, issame_list, thrsh)

        print('acc:{}'.format(accuracy))
        print('tpr:{}'.format(tpr))




