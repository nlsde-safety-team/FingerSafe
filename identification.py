from collections import defaultdict
import torch
import numpy as np
from models.inception_resnet_v1 import ResNet, DenseNet, Inceptionv3
from FingerprintDataset import FingerprintTest, Gray
import cv2
import argparse
from sklearn.metrics import accuracy_score

# https://github.com/ivonajdenkoska/fingerprint-recognition/blob/master/src/fingerprint-recognition.ipynb

def perform_identification_scenario(train_feature, train_label, query_feature, query_label, should_draw):
    true_y = []
    pred_y = []
    total_prob = 0
    # print("----- START, threshold = {}, rank = {} -----".format(dist_threshold, rank))
    # dists = [[(e1 - e2).norm().item() for e2 in train_feature] for e1 in query_feature]
    dists = [[np.linalg.norm(e1 - e2).item() for e2 in train_feature] for e1 in query_feature]
    # print(len(dists))

    for index in range(len(query_label)):
        # Get the distances between the query image and all other training images
        # best_matches_dict = get_best_matches(query_feature[index], train_feature, dist_threshold)
        true_class = query_label[index]

        # Classify the first closest features according to the given rank
        # first_rank_fprs = classify_fpr(dists[index], train_label, rank)
        # predicted_class = first_rank_fprs[0][0]
        # prob = first_rank_fprs[0][1] / TRAIN_PER_CLASS
        # total_prob += prob
        best_index = np.argmin(dists[index])
        predicted_class = train_label[best_index]
        true_y.append(true_class)  # true_class
        pred_y.append(predicted_class)
    # print(true_y)
    # print(pred_y)

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
    # embedding = torch.from_numpy(embedding).to(device)
    return embedding


from kymatio import Scattering2D
def SVM(valid):
    scattering = Scattering2D(J=2, shape=(50, 50))
    # if torch.cuda.is_available():
    #     print("Move scattering to GPU")
    #     scattering = scattering.cuda()

    batch_size = 4
    idx = 0
    embedding_size = 11664#1*81*6*6
    embedding = torch.zeros([len(valid), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(embedding):
            batch = valid[idx:idx + batch_size].clone().detach().cpu().numpy()
            emb = scattering(batch)
            emb = torch.from_numpy(emb)
            embedding[idx:idx + batch_size] = emb.view(emb.size(0), -1)
            idx += batch_size
    return embedding.cpu().numpy()

if __name__ == '__main__':
    """
        For identification, calculating the identification ACC
        - weight_name: the weights of model to load
        - root_adv: the menu of adv sets, organized be like: 4 train / 2 test per class
        - database: subdirectory of data in database set, RGB=training, MHS=2015_train, HG=2017_train, Frangi=frangi_train
        - query: subdirectory of data in query set, RGB=testing, MHS=2015_test, HG=2017_test, Frangi=frangi_test
        - backbone: the kind of method to attack, ['ResNet', 'InceptionV3', 'DenseNet', 'ScatNet']
        """
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', '-w', type=str,
                        help='name of loading model', default="./best_models/clean_split_1009.pth")
    parser.add_argument('--root_adv', '-ra', type=str,
                        help='root of adv data',
                        default="./datasets/final_identification/iden_fingersafe")
    parser.add_argument('--database', '-db', type=str,
                        help='subdirectory of data in database set, RGB=training, MHS=2015_train, HG=2017_train, Frangi=frangi_train',
                        default="training")
    parser.add_argument('--query', '-q', type=str,
                        help='subdirectory of data in query set, RGB=testing, MHS=2015_test, HG=2017_test, Frangi=frangi_test',
                        default="testing")
    parser.add_argument('--backbone', type=str, default='ResNet')
    args = parser.parse_args()

    if args.database != 'training':
        is_grey = True
    else:
        is_grey = False
    grey = Gray()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    weight_name = args.weight

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

    # load 68 classes * 6 images (4 train / 2 test), 4 images for each into "database set", 2 into "query set"
    dataroot_adv = args.root_adv
    dataroot_clean = './datasets/final_identification/iden_clean'
    dataroot_database = args.database
    dataroot_query = args.query
    print(dataroot_clean)
    print(dataroot_adv) 
    database_set = FingerprintTest(dataroot_clean, dataroot_database)  
    query_set = FingerprintTest(dataroot_adv, dataroot_query)

    if is_grey:
        x_data = torch.from_numpy(np.array([(grey(t)).numpy() for t in database_set.x_data]))
        q_data = torch.from_numpy(np.array([(grey(t)).numpy() for t in query_set.x_data]))
    else:
        x_data = torch.from_numpy(np.array([t.numpy() for t in database_set.x_data]))
        q_data = torch.from_numpy(np.array([t.numpy() for t in query_set.x_data]))

    if args.backbone in ['ResNet', 'DenseNet', 'InceptionV3']:
        database_embedding = cal_embed(x_data, model, embedding_size=embedding_size, device=device)  # 272*1000
        query_embedding = cal_embed(q_data, model, embedding_size=embedding_size, device=device)  # 136*1000
    else:
        database_embedding = SVM(x_data)
        query_embedding = SVM(q_data)

    database_label = database_set.y_data.numpy()
    query_label = query_set.y_data.numpy()

    acc = perform_identification_scenario(database_embedding, database_label, query_embedding, query_label, False)

    with open('identification_result.txt', 'a') as f:
        f.write('root_adv: {}, database:{}, query:{}, backbone:{}\n'.format(dataroot_adv, dataroot_database, dataroot_query, args.backbone))
        f.write('acc:{:.4f}\n\n'.format(acc))
