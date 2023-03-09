from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
import torch
import os
from models.inception_resnet_v1 import ResNet, DenseNet, Inceptionv3
from FingerprintDataset import FingerprintTrain
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np


def plot_with_labels(lowDWeights_resnet, lowDWeights_inception, lowDWeights_densenet, labels, i):
    plt.cla()
    # X_clean, X_fingersafe, X_pgd, X_min = lowDWeights_clean[:, 0], lowDWeights_fingersafe[:, 0], \
    #                                       lowDWeights_pgd[:, 0], lowDWeights_min[:, 0]
    # Y_clean, Y_fingersafe, Y_pgd, Y_min = lowDWeights_clean[:, 1], lowDWeights_fingersafe[:, 1], \
    #                                       lowDWeights_pgd[:, 1], lowDWeights_min[:, 1]
    classes = [i]
    index = []
    for iclass in classes:
        index.extend((x + iclass * 6) for x in range(6))
    print(index)
    x_fingersafe_resnet = lowDWeights_resnet[0][:, 0]
    y_fingersafe_resnet = lowDWeights_resnet[0][:, 1]
    x_fingersafe_inception = lowDWeights_inception[0][:, 0]
    y_fingersafe_inception = lowDWeights_inception[0][:, 1]
    x_fingersafe_densenet = lowDWeights_densenet[0][:, 0]
    y_fingersafe_densenet = lowDWeights_densenet[0][:, 1]

    x_pgd_resnet = lowDWeights_resnet[1][:, 0]
    y_pgd_resnet = lowDWeights_resnet[1][:, 1]
    x_pgd_inception = lowDWeights_inception[1][:, 0]
    y_pgd_inception = lowDWeights_inception[1][:, 1]
    x_pgd_densenet = lowDWeights_densenet[1][:, 0]
    y_pgd_densenet = lowDWeights_densenet[1][:, 1]

    x_min_resnet = lowDWeights_resnet[2][:, 0]
    y_min_resnet = lowDWeights_resnet[2][:, 1]
    x_min_inception = lowDWeights_inception[2][:, 0]
    y_min_inception = lowDWeights_inception[2][:, 1]
    x_min_densenet = lowDWeights_densenet[2][:, 0]
    y_min_densenet = lowDWeights_densenet[2][:, 1]

    x_fk_resnet = lowDWeights_resnet[3][:, 0]
    y_fk_resnet = lowDWeights_resnet[3][:, 1]
    x_fk_inception = lowDWeights_inception[3][:, 0]
    y_fk_inception = lowDWeights_inception[3][:, 1]
    x_fk_densenet = lowDWeights_densenet[3][:, 0]
    y_fk_densenet = lowDWeights_densenet[3][:, 1]

    x_clean_resnet = lowDWeights_resnet[4][:, 0]
    y_clean_resnet = lowDWeights_resnet[4][:, 1]
    x_clean_inception = lowDWeights_inception[4][:, 0]
    y_clean_inception = lowDWeights_inception[4][:, 1]
    x_clean_densenet = lowDWeights_densenet[4][:, 0]
    y_clean_densenet = lowDWeights_densenet[4][:, 1]

    # x_clean = [X_clean[i] for i in index]
    # x_fingersafe = [X_fingersafe[i] for i in index]
    # x_pgd = [X_pgd[i] for i in index]
    # x_min = [X_min[i] for i in index]
    # y_clean = [Y_clean[i] for i in index]
    # y_fingersafe = [Y_fingersafe[i] for i in index]
    # y_pgd = [Y_pgd[i] for i in index]
    # y_min = [Y_min[i] for i in index]
    label_fingersafe, label_pgd, label_min, label_fk, label_clean = [], [], [], [], []
    for iclass in range(len(classes)):
        label_fingersafe.extend((iclass + 1) for i in range(12))
        label_pgd.extend((iclass+3) for i in range(6))
        label_min.extend((iclass+5) for i in range(6))
        label_fk.extend((iclass+7) for i in range(6))
        label_clean.extend((iclass + 9) for i in range(6))
    import numpy as np
    # colors = np.zeros(labels.shape)
    # i = 0
    # for x, y, s in zip(X, Y, labels):
    #     c = cm.rainbow(int(255/9 * s))  
    #     plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    # plt.xlim(X.min(), X.max())
    # plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer')
    colormap = cm.rainbow(np.linspace(0, 1, 50))
    # ax1 = plt.subplot(2, 2, 1)
    # plt.scatter(x_clean, y_clean, c=colormap[label], marker='o', label="clean")
    # plt.legend()
    # ax2 = plt.subplot(2, 2, 2)
    # plt.scatter(x_fingersafe, y_fingersafe, c=colormap[label], marker='*', label="fingersafe")
    # plt.legend()
    # ax3 = plt.subplot(2, 2, 3)
    # plt.scatter(x_pgd, y_pgd, c=colormap[label], marker='s', label="pgd")
    # plt.legend()
    # ax4 = plt.subplot(2, 2, 4)
    # plt.scatter(x_min, y_min, c=colormap[label], marker='^', label="min")
    # plt.legend()

    # plt.figure(figsize=(60, 10))
    # plt.subplot(1, 5, 1)
    # plt.scatter(x_clean_inception, y_clean_inception, c=colormap[labels], marker='o', label="clean_inception")
    # plt.subplot(1, 5, 2)
    # plt.scatter(x_pgd_inception, y_pgd_inception, c=colormap[labels], marker='o', label="pgd_inception")
    # plt.subplot(1, 5, 3)
    # plt.scatter(x_min_inception, y_min_inception, c=colormap[labels], marker='o', label="min_inception")
    # plt.subplot(1, 5, 4)
    # plt.scatter(x_fk_inception, y_fk_inception, c=colormap[labels], marker='o', label="fk_inception")
    # plt.subplot(1, 5, 5)
    # plt.scatter(x_fingersafe_inception, y_fingersafe_inception, c=colormap[labels], marker='o', label="fingersafe_inception")


    # plt.figure(figsize=(60, 10))
    # plt.subplot(1, 5, 1)
    # plt.scatter(x_clean_resnet, y_clean_resnet, c=colormap[labels], marker='o', label="clean_resnet")
    # plt.subplot(1, 5, 2)
    # plt.scatter(x_pgd_resnet, y_pgd_resnet, c=colormap[labels], marker='o', label="pgd_resnet")
    # plt.subplot(1, 5, 3)
    # plt.scatter(x_min_resnet, y_min_resnet, c=colormap[labels], marker='o', label="min_resnet")
    # plt.subplot(1, 5, 4)
    # plt.scatter(x_fk_resnet, y_fk_resnet, c=colormap[labels], marker='o', label="fk_resnet")
    # plt.subplot(1, 5, 5)
    # plt.scatter(x_fingersafe_resnet, y_fingersafe_resnet, c=colormap[labels], marker='o', label="fingersafe_resnet")

    plt.figure(figsize=(60, 10))
    plt.subplot(1, 5, 1)
    plt.scatter(x_clean_densenet, y_clean_densenet, c=colormap[labels], marker='o', label="clean_densenet")
    # for x, y, s in zip(x_clean_densenet, y_clean_densenet, labels):
    #     c = cm.rainbow(int(255/20 * s)) 
    #     plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    # plt.xlim(x_clean_densenet.min(), x_clean_densenet.max())
    # plt.ylim(y_clean_densenet.min(), y_clean_densenet.max()); plt.title('clean')
    plt.subplot(1, 5, 2)
    plt.scatter(x_pgd_densenet, y_pgd_densenet, c=colormap[labels], marker='o', label="pgd_densenet")
    # for x, y, s in zip(x_pgd_densenet, y_pgd_densenet, labels):
    #     c = cm.rainbow(int(255/20 * s)) 
    #     plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    # plt.xlim(x_pgd_densenet.min(), x_pgd_densenet.max())
    # plt.ylim(y_pgd_densenet.min(), y_pgd_densenet.max())
    # plt.title('pgd')
    plt.subplot(1, 5, 3)
    plt.scatter(x_min_densenet, y_min_densenet, c=colormap[labels], marker='o', label="min_densenet")
    # for x, y, s in zip(x_min_densenet, y_min_densenet, labels):
    #     c = cm.rainbow(int(255/20 * s))  
    #     plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    # plt.xlim(x_min_densenet.min(), x_min_densenet.max())
    # plt.ylim(y_min_densenet.min(), y_min_densenet.max())
    # plt.title('min')
    plt.subplot(1, 5, 4)
    # plt.scatter(x_fk_densenet, y_fk_densenet, c=colormap[labels], marker='o', label="fk_densenet")
    for x, y, s in zip(x_fk_densenet, y_fk_densenet, labels):
        c = cm.rainbow(int(255/50 * s)) 
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(x_fk_densenet.min(), x_fk_densenet.max())
    plt.ylim(y_fk_densenet.min(), y_fk_densenet.max())
    plt.title('fk')
    plt.subplot(1, 5, 5)
    # plt.scatter(x_fingersafe_densenet, y_fingersafe_densenet, c=colormap[labels], marker='o', label="fingersafe_densenet")
    for x, y, s in zip(x_fingersafe_densenet, y_fingersafe_densenet, labels):
        c = cm.rainbow(int(255/50 * s)) 
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(x_fingersafe_densenet.min(), x_fingersafe_densenet.max())
    plt.ylim(y_fingersafe_densenet.min(), y_fingersafe_densenet.max())
    plt.title('fingersafe')
    plt.legend()

    plt.savefig("./results/2_densenet.jpg".format(classes[0]))

# 类内
def cal_intra_difference(embedding, embedding_adv):
    index = 0
    distance = 0
    image_per_class = 6
    mask = (torch.ones(image_per_class, image_per_class) - torch.eye(image_per_class)).to(device)
    while (index < len(embedding)):
        clean_rep = embedding[index: index + image_per_class]
        class_rep = embedding_adv[index: index + image_per_class]  # here embedding_adv can be embedding_center
        class_rep_dim0 = class_rep.repeat(image_per_class, 1, 1)
        class_rep_dim1 = class_rep.repeat(image_per_class, 1, 1).transpose(0, 1)
        # loss_rep_diff = torch.sum((class_rep_dim0 - class_rep_dim1).abs(), dim=-1).to(device)  # L1
        # loss_rep_diff = torch.sum(torch.pow((class_rep_dim0 - class_rep_dim1), 2), dim=-1).to(device)  # L2
        loss_rep_diff = torch.norm((class_rep_dim0 - class_rep_dim1), dim=-1, p=2)  # 6*6
        print(loss_rep_diff.shape)  # 6*6
        loss_rep_diff = torch.einsum('ij, ij->ij', loss_rep_diff, mask)
        distance += torch.sum(loss_rep_diff) / (image_per_class * (image_per_class-1)) 
        # images_rep_dim1 = clean_rep.repeat(image_per_class, 1, 1).transpose(0, 1)
        # loss_rep = F.pairwise_distance(class_rep_dim0, images_rep_dim1, p=2)
        # loss_rep = torch.norm((class_rep_dim0 - images_rep_dim1), dim=-1, p=2)  # 6*6
        # distance += torch.mean(loss_rep)
        index += image_per_class
    print(distance)


# 类间
def cal_inter_difference(embeddings, embeddings_clean):
    index = 0
    distance = 0
    image_per_class = 6
    mask = (torch.ones(image_per_class, image_per_class) - torch.eye(image_per_class)).to(device)
    while index < len(embeddings):
        class_rep = embeddings[index: index + image_per_class]  # here embeddings can be embedding_center
        class_rep_dim0 = class_rep.repeat(image_per_class, 1, 1)
        class_rep_dim1 = class_rep.repeat(image_per_class, 1, 1).transpose(0, 1)

        # for iclass in range(68):  # 68 classes
        #     if iclass == index // image_per_class:
        #         continue
        #     other_rep = embeddings[iclass * image_per_class: (iclass + 1) * image_per_class]
        #     other_rep_dim1 = other_rep.repeat(image_per_class, 1, 1).transpose(0, 1)
        #
        #    
        #     loss_rep_diff = torch.sum(torch.pow((class_rep_dim0 - other_rep_dim1), 2), dim=-1).to(device)  # L2
        #     loss_rep_diff = torch.einsum('ij, ij->ij', loss_rep_diff, mask)
        #     distance += torch.sum(loss_rep_diff) / (image_per_class * (image_per_class - 1)) 
        clean_rep = embeddings_clean[index: index + image_per_class]
        images_rep_dim1 = clean_rep.repeat(image_per_class, 1, 1).transpose(0, 1)
        loss_rep = F.pairwise_distance(class_rep_dim0, images_rep_dim1, p=2)
        distance += torch.mean(loss_rep)
        index += image_per_class
    print(distance) 


def inter_class(embedding):
    index = 0
    distance = 0
    image_per_class = 6
    centers = torch.zeros((68, 1000))
    mask = (torch.ones(image_per_class, image_per_class) - torch.eye(image_per_class)).to(device)
    while (index < len(embedding)):
        rep = embedding[index: index + 6]  # 6*1000
        embedding_center = torch.mean(rep, dim=0)  # 1*1000
        embedding_std = torch.std(rep, dim=0)  # 1*1000
        distance += torch.mean(embedding_std)
        centers[index // 6] = embedding_center
        index += 6
    print(distance / 68)
    cal_difference(centers, centers)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    plot_only = 300  # 50 classes

    resnet = ResNet(
        classify=False,
        nclasses=268
    ).to(device)
    inception = Inceptionv3(classify=False, nclasses=268).to(device)
    densenet = DenseNet(classify=False, nclasses=268).to(device)
    resnet.load_state_dict(torch.load('./best_models/fk_split_best.pth'))
    inception.load_state_dict(torch.load('./best_models/inception_clean_best.pth'))
    densenet.load_state_dict(torch.load('./best_models/densenet_clean_best.pth'))
    inception.eval()
    densenet.eval()
    resnet.eval()

    # tsne_clean = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000) 
    tsne_fingersafe = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000) 
    tsne_pgd = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000) 
    tsne_min = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000) 
    tsne_fk = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    # pca = PCA(n_components=2)


    clean_set = FingerprintTrain('./datasets/final/veri_test', '')
    clean_loader = DataLoader(dataset=clean_set, batch_size=plot_only, shuffle=False)
    fingersafe_set = FingerprintTrain('./datasets/final/veri_sample_fingersafe', '')
    fingersafe_loader = DataLoader(dataset=fingersafe_set, batch_size=plot_only, shuffle=False)
    pgd_set = FingerprintTrain('./datasets/final/sample_pgd', '')
    pgd_loader = DataLoader(dataset=pgd_set, batch_size=plot_only, shuffle=False)
    min_set = FingerprintTrain('./datasets/final/sample_min', '')
    min_loader = DataLoader(dataset=min_set, batch_size=plot_only, shuffle=False)
    fk_set = FingerprintTrain('./datasets/final/sample_fk', '')
    fk_loader = DataLoader(dataset=fk_set, batch_size=plot_only, shuffle=False)

    # for index in range(4):
    #     if index==0:
    #         loader = clean_loader
    #         print('clean')
    #     elif index==1:
    #         loader = fingersafe_loader
    #         print('fingersafe')
    #     elif index==2:
    #         loader = pgd_loader
    #         print('pgd')
    #     elif index==3:
    #         loader = min_loader
    #         print('min')
    #
    #
    #     with torch.no_grad():
    #         for i, (data, target) in enumerate(clean_loader):
    #             data = data.to(device)
    #             embedding = resnet(data)
    #             batch_labels = target
    #             break
    #     with torch.no_grad():
    #         for i, (data, target) in enumerate(loader):
    #             data = data.to(device)
    #             embedding_adv = resnet(data)
    #             batch_labels = target
    #             break
    # with torch.no_grad():
    #     for i, (data, target) in enumerate(clean_loader):
    #         data = data.to(device)
    #         embedding = resnet(data)
    #         batch_labels = target
    #         break
    emb_resnet, emb_inc, emb_den = [], [], []
    np.random.seed(0)
    with torch.no_grad():
        for i, (data, target, _) in enumerate(fingersafe_loader):
            print('fingersafe')
            print(data.shape)
            data = data.to(device)
            embedding_fingersafe_resnet = resnet(data)
            embedding_fingersafe_inception = inception(data)
            embedding_fingersafe_densenet = densenet(data)
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_resnet.append(tsne.fit_transform(embedding_fingersafe_resnet.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_inc.append(tsne.fit_transform(embedding_fingersafe_inception.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_den.append(tsne.fit_transform(embedding_fingersafe_densenet.data.cpu().numpy()[:plot_only, :]))
            if i == 0:
                batch_labels = target
            if i == 0:
                break
        for i, (data, target, _) in enumerate(pgd_loader):
            print('pgd')
            data = data.to(device)
            embedding_pgd_resnet = resnet(data)
            embedding_pgd_inception = inception(data)
            embedding_pgd_densenet = densenet(data)
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_resnet.append(tsne.fit_transform(embedding_pgd_resnet.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_inc.append(tsne.fit_transform(embedding_pgd_inception.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_den.append(tsne.fit_transform(embedding_pgd_densenet.data.cpu().numpy()[:plot_only, :]))
            if i == 0:
                break
        for i, (data, target, _) in enumerate(min_loader):
            print('min')
            data = data.to(device)
            embedding_min_resnet = resnet(data)
            embedding_min_inception = inception(data)
            embedding_min_densenet = densenet(data)
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_resnet.append(tsne.fit_transform(embedding_min_resnet.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_inc.append(tsne.fit_transform(embedding_min_inception.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_den.append(tsne.fit_transform(embedding_min_densenet.data.cpu().numpy()[:plot_only, :]))
            if i == 0:
                break
        for i, (data, target, _) in enumerate(fk_loader):
            print('fk')
            data = data.to(device)
            embedding_fk_resnet = resnet(data)
            embedding_fk_inception = inception(data)
            embedding_fk_densenet = densenet(data)
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_resnet.append(tsne.fit_transform(embedding_fk_resnet.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_inc.append(tsne.fit_transform(embedding_fk_inception.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_den.append(tsne.fit_transform(embedding_fk_densenet.data.cpu().numpy()[:plot_only, :]))
            if i == 0:
                break
        for i, (data, target, _) in enumerate(clean_loader):
            print('clean')
            data = data.to(device)
            embedding_clean_resnet = resnet(data)
            embedding_clean_inception = inception(data)
            embedding_clean_densenet = densenet(data)
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_resnet.append(tsne.fit_transform(embedding_clean_resnet.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_inc.append(tsne.fit_transform(embedding_clean_inception.data.cpu().numpy()[:plot_only, :]))
            tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)
            emb_den.append(tsne.fit_transform(embedding_clean_densenet.data.cpu().numpy()[:plot_only, :]))
            if i == 0:
                break

    # calculate inner-class difference
    # cal_difference(embedding, embedding_fingersafe)
    # cal_inter_difference(embedding_pgd_densenet, embedding_clean_densenet)
    # cal_inter_difference(embedding_min_densenet, embedding_clean_densenet)
    # cal_inter_difference(embedding_fk_densenet, embedding_clean_densenet)
    # cal_inter_difference(embedding_fingersafe_densenet, embedding_clean_densenet)
    # exit(0)


    # low_dim_embs_resnet = tsne_clean.fit_transform(embedding_resnet.data.cpu().numpy()[:plot_only, :])
    # low_dim_embs_fingersafe = tsne_fingersafe.fit_transform(embedding_fingersafe.data.cpu().numpy()[:plot_only, :])
    # low_dim_embs_pgd = tsne_pgd.fit_transform(embedding_pgd.data.cpu().numpy()[:plot_only, :])
    # low_dim_embs_min = tsne_min.fit_transform(embedding_min.data.cpu().numpy()[:plot_only, :])
    labels = batch_labels.numpy()[:plot_only]
    print(labels)
    for i_class in range(68):
        plot_with_labels(emb_resnet, emb_inc, emb_den, labels, i_class)
        exit(0)
