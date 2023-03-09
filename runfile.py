"""
  Purpose:  Train and save network weights
"""
import argparse
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

    model = Model(classify=True, nclasses=268).to(device)
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

        iden_acc = identify(database_set, query_set, model, embedding_size)
        print('clean iden_acc = {}'.format(iden_acc))
        emb_clean1, emb_clean2 = cal_embed(clean_1, clean_2, model, embedding_size=embedding_size, device=device)
        _, veri_tpr, veri_acc, _ = evaluate_lsm(201, emb_clean1, emb_clean2, issame_clean.to(device), embedding_size)
        print('clean veri_acc = {}, clean_veri_tpr={}'.format(veri_acc, veri_tpr))

        # save the parameters of the model according to the best performence of acc in verification
        if accuracy >= best_acc and epoch > 2:
            best_acc = accuracy
            file_name = '{}_best.pth'.format(type)
            torch.save(model.state_dict(), os.path.join("saved_models", file_name))
            print("Saved: ", file_name)



def main():
    """
    Train a network that can identify and verify the fingerprints
    train_path: the path of training sets
    test_pkl: the path of the map which organizes the testing set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int,
                        help='batch size', default=4)
    parser.add_argument('--epochs', '-e', type=int,
                        help='the number of epoch for training process', default=30)
    parser.add_argument('--adv', '-a', type=str,
                        help='type of adv data',
                        default="clean")
    parser.add_argument('--train_path',  type=str,
                        help='path of training data',
                        default="./datasets/final/clean_split")
    parser.add_argument('--test_pkl', type=str,
                        help='path of testing data',
                        default="./datapaths/datapath_valid_clean_test.pkl")
    args = parser.parse_args()

    batch_size = args.batch_size  # 4
    epochs = args.epochs  # 30
    adv = args.adv  # 'fingersafe'  # clean_split = train_split
    train_path = args.train_path
    test_pkl = args.test_pkl
    valid_1, valid_2, issame_list, _, _ = get_valid_data(test_pkl, flag=True)
    clean_1, clean2, issame_clean, _, _ = get_valid_data("./datapaths/datapath_clean_test.pkl")
    clean_dataroot = './datasets/final_identification/iden_clean'
    adv_dataroot = './datasets/final_identification/iden_clean'
    database_set = FingerprintTest(clean_dataroot, 'training')  # keep clean
    query_set = FingerprintTest(adv_dataroot, 'testing')  # keep clean
    train_all(adv, device, train_path, batch_size, epochs,
                                                valid_1, valid_2, issame_list, clean_1, clean2, issame_clean,
                                                database_set, query_set)


if __name__ == '__main__':
    main()

