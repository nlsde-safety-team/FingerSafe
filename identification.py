from collections import defaultdict
import torch
import numpy as np
from models.inception_resnet_v1 import ResNet as ResNet
from FingerprintDataset import FingerprintTest
import cv2
import operator
from sklearn.metrics import accuracy_score

# identification
def perform_identification_scenario(train_feature, train_label, query_feature, query_label, should_draw):
    true_y = []
    pred_y = []
    total_prob = 0
    dists = [[np.linalg.norm(e1 - e2).item() for e2 in train_feature] for e1 in query_feature]

    for index in range(len(query_label)):
        # Get the distances between the query image and all other training images
        true_class = query_label[index]
        best_index = np.argmin(dists[index])
        predicted_class = train_label[best_index]
        true_y.append(true_class)  # true_class
        pred_y.append(predicted_class)

    print("Accuracy is %f " % (accuracy_score(true_y, pred_y)))
    return accuracy_score(true_y, pred_y)


def cal_embed(data, model, embedding_size, device):
    batch_size = 4
    idx = 0
    embedding = np.zeros([len(data), embedding_size])
    model.classify = False
    model.eval()
    with torch.no_grad():
        while idx + batch_size <= len(embedding):
            batch = data[idx:idx + batch_size].clone().detach()
            embedding[idx:idx + batch_size] = model(batch.to(device)).cpu()
            idx += batch_size
    return embedding


from kymatio import Scattering2D
def SVM(valid):
    scattering = Scattering2D(J=2, shape=(50, 50))
    if torch.cuda.is_available():
        print("Move scattering to GPU")
        scattering = scattering.cuda()

    batch_size = 4
    idx = 0
    embedding_size = 11664#1*81*6*6
    embedding = np.zeros([len(valid), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(embedding):
            batch = valid[idx:idx + batch_size].clone().detach()
            emb = scattering(batch.cuda()).cpu()
            embedding[idx:idx + batch_size] = emb.view(emb.size(0), -1)
            idx += batch_size
    return embedding


def identify(database_set, query_set, resnet, size=1000):
    x_data = torch.from_numpy(np.array([t.numpy() for t in database_set.x_data]))
    q_data = torch.from_numpy(np.array([t.numpy() for t in query_set.x_data]))
    database_embedding = cal_embed(x_data, resnet, embedding_size=size, device='cuda')  # 272*1000
    query_embedding = cal_embed(q_data, resnet, embedding_size=size, device='cuda')  # 136*1000
    database_label = database_set.y_data.numpy()
    query_label = query_set.y_data.numpy()

    acc = perform_identification_scenario(database_embedding, database_label, query_embedding, query_label, False)
    return acc


if __name__ == '__main__':
    """
        used as identification, calculate identification ACC
        change the following parameters when using this code
        - weight_name: position of trained model
        - adv_type：test the type of protection technique
        - clean_dataroot: folder of clean data. 4 train / 2 test for each class for HKPolyU
        - adv_dataroot: folder of adv data. 4 train / 2 test for each class for HKPolyU
        - embedding_size：embedding size of different backbone. 1000 for resnet.
        - can use scatnet for feature extraction, then use it for identification.
        """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    weight_name = './best_models/MHS_clean_best_1030.pth'
    resnet = ResNet(
        classify=False,
        nclasses=268
    ).to(device)
    resnet.load_state_dict(torch.load(weight_name))
    print(weight_name)
    resnet.eval()
    adv_type = ['fingersafe']
    for a in adv_type:
        # load 68 classes * 6 images (4 train / 2 test), 4 images for each into database, 2 into query
        clean_dataroot = './datasets/final_identification/iden_clean'  # 保持不变
        adv_dataroot = './datasets/final_identification/iden_{}'.format(a)
        print(clean_dataroot)
        print(adv_dataroot)
        database_set = FingerprintTest(clean_dataroot, '2015_train')  # training set for clean(68*4) as database
        query_set = FingerprintTest(adv_dataroot, '2015_test')  # testing set for adv (68*4) as query

        identify(database_set, query_set, resnet, size=1000)
