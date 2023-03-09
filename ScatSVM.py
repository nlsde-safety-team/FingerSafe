import numpy as np
from verification import get_valid_data
import torch
from FingerprintDataset import FingerprintTest
from torch.utils.data import Dataset, DataLoader
from kymatio import Scattering2D
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
def Scat(valid_1, valid_2, size):
    scattering = Scattering2D(J=2, shape=(24, 24))
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
            batch = valid_1[idx:idx + batch_size].clone().detach().cpu().numpy()
            emb = scattering(batch).cpu()
            embedding_1[idx:idx + batch_size] = emb.view(emb.size(0), -1)
            batch = valid_2[idx:idx + batch_size].clone().detach().cpu().numpy()
            emb = scattering(batch).cpu()
            embedding_2[idx:idx + batch_size] = emb.view(emb.size(0), -1)
            idx += batch_size

    embedding_1 = torch.from_numpy(embedding_1).to(device)
    embedding_2 = torch.from_numpy(embedding_2).to(device)
    return embedding_1, embedding_2


def SVM(train_1, train_2, issame_train,
        test_1, test_2, issame_test):
    clf_svm = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr')
    clf_rdf = RandomForestClassifier(n_estimators=100,
                                     bootstrap=True,
                                     max_features='sqrt')
    emb = (train_1 - train_2).cpu().numpy()
    issame = issame_train.numpy().ravel()
    clf_svm.fit(emb, issame)
    print('SVM classifier training')
    print(clf_svm.score(emb, issame))
    print('RDF classifier training')
    clf_rdf.fit(emb, issame)
    print(clf_rdf.score(emb, issame))
    emb_test = (test_1 - test_2).cpu().numpy()
    issame_test = issame_test.numpy().ravel()
    y_hat = clf_svm.predict(emb_test)
    print('SVM predict_test :\n', clf_svm.score(emb_test, issame_test))
    print(y_hat)
    y_hat = clf_rdf.predict(emb_test)
    print('RDF predict_test :\n', clf_rdf.score(emb_test, issame_test))
    print(y_hat)


def SVM_iden(train_loader, test_loader):
    scattering = Scattering2D(J=2, shape=(24, 24))
    clf_svm = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr')
    clf_rdf = RandomForestClassifier(n_estimators=100,
                                        bootstrap=True,
                                        max_features='sqrt')
    if torch.cuda.is_available():
        print("Move scattering to GPU")
        scattering = scattering.cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        x_train = scattering(data.cuda()) 
        x_train = x_train.view(x_train.size(0), -1)
        x_train = x_train.cpu().numpy()
        print('SVM classifier training')
        clf_svm.fit(x_train, target.numpy().ravel())
        print(clf_svm.score(x_train, target))
        print('RDF classifier training')
        clf_rdf.fit(x_train, target.numpy().ravel())
        print(clf_rdf.score(x_train, target))

    for i, (data, target) in enumerate(test_loader):
        x_test = scattering(data.cuda())
        x_test = x_test.view(x_test.size(0), -1)
        x_test = x_test.cpu().numpy()
        y_hat = clf_svm.predict(x_test)
        print('SVM predict_test :\n', clf_svm.score(x_test, target))
        y_hat = clf_rdf.predict(x_test)
        print('RDF predict_test :\n', clf_rdf.score(x_test, target))

def main():
    train_root = './datapaths/datapath_physical_MHS_database_train.pkl' 
    test_root = './datapaths/datapath_physical_MHS_evaluation_test.pkl'
    test_adv_root = './datapaths/datapath_physical_MHS_after_fingersafe_test.pkl'
    train_1, train_2, issame_train = get_valid_data(train_root, num=30)
    test_1, test_2, issame_test = get_valid_data(test_root, test_adv_root, num=20)
    train_1, train_2 = Scat(train_1, train_2, size=1*81*6*6)
    test_1, test_2 = Scat(test_1, test_2, size=1*81*6*6)
    SVM(train_1, train_2, issame_train, test_1, test_2, issame_test)
    # train_root = './datasets/final_identification/iden_clean'
    # test_root = './datasets/final_identification/iden_clean'
    # print(train_root)
    # print(test_root)
    # train_set = FingerprintTest(train_root, '2015_train')
    # train_loader = DataLoader(dataset=train_set,
    #                          batch_size=272,
    #                          shuffle=False)
    # test_set = FingerprintTest(test_root, '2015_test')
    # test_loader = DataLoader(dataset=test_set,
    #                          batch_size=136,
    #                          shuffle=False)
    # SVM_iden(train_loader, test_loader)


if __name__ == '__main__':
    main()
