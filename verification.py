#%% md

### facenet-pytorch LFW evaluation
# This notebook demonstrates how to evaluate performance against the LFW dataset.

#%%

# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os
from models.inception_resnet_v1 import ResNet, DenseNet, Inceptionv3
from data import DataSampler,  DataSampler_adv
import matplotlib.pyplot as plt
import argparse

#### Evaluate embeddings by using distance metrics to perform verification on the official LFW veri_test set.
"""
The functions in the next block are copy pasted from `facenet.src.lfw`. Unfortunately that module has an absolute import from `facenet`, so can't be imported from the submodule

added functionality to return false positive and false negatives
"""
#%%

from sklearn.model_selection import KFold
from scipy import interpolate
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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
    gen_plot(fpr, 1 - recall)
    # print(True_Positive[margin_index], True_Negative[margin_index], False_Positive[margin_index], False_Negative[margin_index])
    print(avg_loss / iterations, tpr, F1_score_max, precision_max, recall_max, accuracy_max, threshold_final)
    # print("acc={}, tpr={}, fpr={}, precision={}, F1_score={}, recall={}"
    #       .format(accuracy_max, tpr, fpr, precision_max, F1_score_max, recall_max))
    return avg_loss / iterations, tpr, accuracy_max, threshold_final

def evaluate_save(iterations, embeddings1, embeddings2, issame, embedding_size=1000, method='fingersafe'):
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

    tpr = True_Positive / (iterations * batch / 2)
    fpr = False_Positive / (iterations * batch / 2)
    tnr = True_Negative / (iterations * batch / 2)
    fnr = False_Negative / (iterations * batch / 2)
    np.save('./roc_det/{}_tpr.npy'.format(method), tpr.detach().cpu().numpy())
    np.save('./roc_det/{}_fpr.npy'.format(method), fpr.detach().cpu().numpy())
    np.save('./roc_det/{}_tnr.npy'.format(method), tnr.detach().cpu().numpy())
    np.save('./roc_det/{}_fnr.npy'.format(method), fnr.detach().cpu().numpy())


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
        with open('verification_result.txt', 'a') as f:
            f.write('veri_test loss:{:.4f}, threshold:{:.4f}, tpr:{:.4f}, acc:{:.4f}\n\n'.format(float(avg_loss / iterations),
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


def gen_plot(fpr, fnr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("FNR", fontsize=14)
    plt.title("ROC Curve-clean", fontsize=14)
    plot = plt.plot(fpr, fnr, linewidth=2)
    plt.savefig('ROC_clean_test.jpg')
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
    if pkl_path_adv != None:
        print(ad2[0])

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
    model.classify = False
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


def Scat(valid_1, valid_2, size=1*81*12*12):
    from kymatio import Scattering2D
    scattering = Scattering2D(J=2, shape=(50, 50))
    # if torch.cuda.is_available():
    #     print("Move scattering to GPU")
    #     scattering = scattering.cuda()
    #     device = 'cuda'

    batch_size = 4
    idx = 0
    embedding_size = size
    embedding_1 = torch.zeros([len(valid_1), embedding_size])
    embedding_2 = torch.zeros([len(valid_2), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(embedding_1):
            batch = valid_1[idx:idx + batch_size].clone().detach().cpu().numpy()
            emb = scattering(batch)
            emb = torch.from_numpy(emb)
            embedding_1[idx:idx + batch_size] = emb.view(emb.size(0), -1)
            batch = valid_2[idx:idx + batch_size].clone().detach().cpu().numpy()
            emb = scattering(batch)
            emb = torch.from_numpy(emb)
            embedding_2[idx:idx + batch_size] = emb.view(emb.size(0), -1)
            idx += batch_size
    embedding_1 = embedding_1.to(device)
    embedding_2 = embedding_2.to(device)
    return embedding_1, embedding_2


if __name__ == '__main__':
    """
    For verification, calculating the verification TPR
    - weight_name: the saved parameters of the model to load
    - backbone: the backbone to evaluate, selecting from ['ResNet', 'InceptionV3', 'DenseNet', 'ScatNet']
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', '-w', type=str,
                        help='name of loading model', default="./best_models/clean_split_1009.pth")
    parser.add_argument('--root_clean', '-rc', type=str,
                        help='root of clean data',
                        default="./datapaths/datapath_clean_test.pkl")
    parser.add_argument('--root_adv', '-ra', type=str,
                        help='root of adv data',
                        default="./datapaths/datapath_fingersafe_test.pkl")
    parser.add_argument('--backbone', '-b', type=str, default='ResNet')
    parser.add_argument('--method', type=str, default='clean')
    args = parser.parse_args()
    with open('verification_result.txt', 'a') as f:
        f.write('root_adv: ' + args.root_adv + ', backbone: ' + args.backbone + '\n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    if args.backbone == 'ResNet':
        model = ResNet( 
            classify=False,
            nclasses=268
        ).to(device)
        embedding_size = 1000
    elif args.backbone == 'DenseNet':
        model = DenseNet(
            classify=False, 
            nclasses=268
        ).to(device)
        embedding_size = 1024
    elif args.backbone == 'InceptionV3':
        model = Inceptionv3(
            classify=False,
            nclasses=268
        ).to(device)
        embedding_size = 2048
    if args.backbone in ['ResNet', 'DenseNet', 'InceptionV3']:
        weight_name = args.weight
        print(weight_name)
        model.load_state_dict(torch.load(weight_name))
        model.eval()

    root_clean = args.root_clean  # str, clean
    clean_1, clean_2, issame_clean, _, _ = get_valid_data(root_clean, num=68)

    root_adv = args.root_adv  # adv
    valid_1, valid_2, issame_list, _, _ = get_valid_data(root_clean, root_adv, num=68)
    print(clean_1.shape, clean_2.shape, valid_1.shape, valid_2.shape)

    if args.backbone in ['ResNet', 'DenseNet', 'InceptionV3']:
        embedding1, embedding2 = cal_embed(valid_1, valid_2, model, embedding_size=embedding_size, device=device)
        emb_clean1, emb_clean2 = cal_embed(clean_1, clean_2, model, embedding_size=embedding_size, device=device)
    elif args.backbone == 'ScatNet':
        embedding1, embedding2 = Scat(valid_1, valid_2, size=11664)
        emb_clean1, emb_clean2 = Scat(clean_1, clean_2, size=11664)

    # # evaluate_save(201, emb_clean1, emb_clean2, issame_clean.to(device))
    # evaluate_save(201, embedding1, embedding2, issame_list.to(device), method=args.method)
    # exit(0)

    eval_loss, tpr, accuracy, thrsh = evaluate_lsm(201, emb_clean1, emb_clean2, issame_clean.to(device))
    test_final(201, embedding1, embedding2, issame_list, thrsh)

    print('acc:{}'.format(accuracy))
    print('tpr:{}'.format(tpr))




