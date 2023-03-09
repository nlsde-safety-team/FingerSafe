import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator
from FingerprintDataset import Fingerprint_Seg
from sklearn.metrics import accuracy_score, precision_score, f1_score
import fingerprint_feature_extractor
from scipy import spatial
from verification import get_valid_data
from fingernet import deploy
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"


# Initiate ORB detector for matching keypoints
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# Extract minutiae keypoints
def extract_kp(image):
    keypoints = []
    FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(image, showResult=False, spuriousMinutiaeThresh=10)
    for termination in FeaturesTerminations:
        keypoints.append(cv2.KeyPoint(termination.locY, termination.locX, 1))
    for bifuication in FeaturesBifurcations:
        keypoints.append(cv2.KeyPoint(bifuication.locY, bifuication.locX, 0))
    # Define descriptor
    orb = cv2.ORB_create()
    # Compute descriptors
    # image = image[:, :, np.newaxis]
    # print(image.shape)
    _, des = orb.compute(image, keypoints)
    return des

def mnt_distance(y_true, y_pred):
    if y_pred.shape[0]==0 or y_true.shape[0]==0:
        return np.pi
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    dis = spatial.distance.cdist(y_pred[:, :2], y_true[:, :2], 'euclidean')
    idx = dis.argmin(axis=1)
    angle = abs(np.mod(y_pred[:,2],2*np.pi) - np.mod(y_true[idx,2],2*np.pi))
    angle = np.asarray([angle, 2*np.pi-angle]).min(axis=0)
    angle = np.asarray([angle, np.pi-angle]).min(axis=0)
    ori = np.mean(angle)
    return ori

def extract_fingernet_minutiae(path):
    return deploy(path)
    # return np.load(path.replace('.bmp', '.npy'))

# Returns feature descriptors for all images from the dataset
def get_feature_descriptors(dataset):
    feature_descriptors = {}
    for image_id, (image, image_label) in enumerate(dataset):
        #kp, des = orb.detectAndCompute(image, None)  # todo use ORB descriptor
        des = extract_kp(image)  # todo use minutiae
        feature_descriptors[image_id] = des
    return feature_descriptors


# Returns best_matches between training features descriptors and query image
def get_best_matches(query_image, trained_features, distance_threshold):
    best_matches_dict = {}
    #kp1, query_des = orb.detectAndCompute(query_image, None)  # features of the query image todo use ORB
    query_des = extract_kp(query_image)  # todo use minutiae
    if query_des is None:
        return None
    for train_image_id in trained_features:
        trained_feature_des = trained_features[train_image_id]
        if query_des is not None and trained_feature_des is not None:
            matches = bf.match(query_des, trained_feature_des)
            # print([m.distance for m in matches])
            matches = sorted(matches, key=lambda x: x.distance, reverse=False)  # sort matches based on feature distance

            best_matches = [m.distance for m in matches if m.distance < distance_threshold]
            best_matches_dict[train_image_id] = len(
                best_matches)  # matching function = length of best matches to given threshold
    best_matches_dict = sorted(best_matches_dict.items(), key=operator.itemgetter(1),
                               reverse=True)  # sort by value - feature distance
    #print('-------------')
    #print(best_matches_dict)
    #print('-------------')
    return best_matches_dict

# Apply homography to test and train image
# Homography or image alignment: to perfectly line up the features in two images
def apply_homography(query_image, closest_image):
    kp1, des1 = orb.detectAndCompute(query_image, None)
    kp2, des2 = orb.detectAndCompute(closest_image, None)
    matches = bf.match(des1, des2)

    # Apply homography
    numGoodMatches = int(len(matches) * 0.5)
    matches = matches[:numGoodMatches]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # M matrix that represents the homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use homography
    height, width = query_image.shape[:2]
    # The function warpPerspective transforms the source image using the specified matrix
    im1Reg = cv2.warpPerspective(closest_image, M, (width, height))

    # Plot aligned query and train image
    plt.subplot(1, 2, 1)
    plt.imsave('./results/train.bmp', im1Reg, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imsave('./results/test.bmp', query_image, cmap='gray')
    # plt.show()



def draw_keypoints_matches(fpr1, fpr2):
    kp1, des1 = orb.detectAndCompute(fpr1, None)
    kp2, des2 = orb.detectAndCompute(fpr2, None)

    matches = bf.match(des1, des2)
    matches.sort(key=lambda x: x.distance, reverse=False)
    imMatches = cv2.drawMatches(fpr1, kp1, fpr2, kp2,matches[:10], None)

    plt.imsave('./results/match.bmp', imMatches)
    plt.show()

def draw_minutiae(image, minutiae, fname, r=15):
    image = np.squeeze(image)
    fig = plt.figure()
    plt.imshow(image,cmap='gray')
    plt.plot(minutiae[:, 0], minutiae[:, 1], 'rs', fillstyle='none', linewidth=1)
    for x, y, o in minutiae:
        plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'r-')
    plt.axis([0,image.shape[1],image.shape[0],0])
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches = 0)
    plt.close(fig)
    return


def count_same_fprs(feature_distances, len_best_matches):
    '''
    Counts how many fprs are close to the query fpr
    :param feature_distances: Feature distaces from given query fpr to all training fprs
    :param len_best_matches: Predefined value for the length of best features
    :return count_same: number of same fprs paris within the given len_best_matches
    '''
    count_same = 0
    for features in feature_distances:
        if int(features[
                   1]) > len_best_matches:  # Compare the len of best features for the given feature with the predefined len
            count_same += 1

    return count_same


def match(des1, des2, score_threshold=0):
    # Todo Matching the differences between the minutiaes extracted by FingerNet
    return mnt_distance(des1, des2) + mnt_distance(des2, des1)
    # Todo Matching between descriptors
    # Brute force match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)
    score = 0
    for match in matches:
        score += match.distance
    if len(matches) == 0:
        avg = np.inf
    else:
        avg = score / len(matches)
    return avg
    if avg < score_threshold:
        return True, avg
    else:
        return False, avg


# Definition of identification scenario
def perform_identification_scenario(train_set, train_feature_descriptors, test_set, dist_threshold, rank):
    true_y = []
    pred_y = []
    total_prob = 0
    print("----- START, threshold = {}, rank = {} -----".format(dist_threshold, rank))
    index = 0
    for (test_image, test_label) in test_set:
        #print('###### {} #########'.format(index))
        index += 1
        test_label = test_label.numpy()
        # Get the distances between the query image and all other training images
        best_matches_dict = get_best_matches(test_image, train_feature_descriptors, dist_threshold)
        if best_matches_dict is None:
            continue
        true_class = test_label

        # Classify the first closest features according to the given rank
        fpr_name, distance = best_matches_dict[0]
        predicted_class = train_set.y_data[fpr_name]
        #print('{}->{}'.format(true_class, predicted_class))
        # first_rank_fprs = classify_fpr(best_matches_dict, rank)
        # predicted_class = first_rank_fprs[0][0]
        # prob = first_rank_fprs[0][1] / TRAIN_PER_CLASS
        # total_prob += prob
        true_y.append(true_class)  # true_class
        pred_y.append(predicted_class)

    avg_probability = total_prob / len(test_set)
    print("Averaged probability for rank %d and threshold %d is %f " % (rank, dist_threshold, avg_probability))
    print("Accuracy for rank %d and threshold %d is %f " % (rank, dist_threshold, accuracy_score(true_y, pred_y)))
    return avg_probability

def perform_identification_scenario_minutiae(train_feature, train_label, query_feature, query_label):
    true_y = []
    pred_y = []
    dists = [[match(e2, e1) for e2 in train_feature] for e1 in query_feature]
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

def identify(name):
    clean_dataroot = './datasets/final_identification/iden_clean'
    adv_dataroot = './datasets/final_identification/iden_{}'.format(name)
    print(clean_dataroot)
    print(adv_dataroot)
    train_set = Fingerprint_Seg(clean_dataroot, '2015_train')
    test_set = Fingerprint_Seg(adv_dataroot, '2015_test')
    print(len(train_set))
    print(len(test_set))
    train_feature_descriptors = get_feature_descriptors(train_set)
    print(len(train_feature_descriptors))

    rank = 1
    for dist_threshold in np.arange(64, 70, 2):
        perform_identification_scenario(train_set, train_feature_descriptors, test_set, dist_threshold, rank)

def identify_minutiae(name):
    clean_dataroot = './datasets/final_identification/iden_clean'
    adv_dataroot = './datasets/final_identification/iden_{}'.format(name)
    print(clean_dataroot)
    print(adv_dataroot)
    train_set = Fingerprint_Seg(clean_dataroot, '2015_train')
    query_set = Fingerprint_Seg(adv_dataroot, '2015_test')
    
    train_label = []
    query_label = []
    train_feature = []
    query_feature = []
    i = 1
    for (path, l) in train_set.paths:
        print('{}/{} train'.format(i, len(train_set.paths)))
        i += 1
        train_feature.append(deploy(path))
        train_label.append(l)

    i = 1
    for (path, l) in query_set.paths:
        print('{}/{} query'.format(i, len(query_set.paths)))
        i += 1
        query_feature.append(deploy(path))
        query_label.append(l)
    acc = perform_identification_scenario_minutiae(train_feature, train_label, query_feature, query_label)
    with open('identification_result.txt', 'a') as f:
        f.write('fingernet: clean-{}, acc:{:.4f}\n\n'.format(name, acc))

def draw_results(results):
    fig = plt.figure()
    for (result, issame) in results:
        if issame:
            plt.scatter(result, 1, c = 'r')
        else:
            plt.scatter(result, 0, c = 'b')
    plt.savefig('ditribution.png')
    plt.close(fig)

def verify():
    root = './datapaths/datapath_MHS_clean_test.pkl'

    valid_1, valid_2, issame_list, add1, add2 = get_valid_data(root, num=68)
    issame_list = np.array(issame_list, dtype=bool)[:, 0]
    results = []
    for idx in range(len(add1)):
        for i in range(8):
            print(str(idx) + '-' + str(i))
            path1 = add1[idx][i]
            path2 = add2[idx][i]
            # data1 = cv2.imread(add1[idx][i], cv2.IMREAD_GRAYSCALE)
            # data2 = cv2.imread(add2[idx][i], cv2.IMREAD_GRAYSCALE)
            issame = not issame_list[idx * 8 + i]
            # des1 = extract_kp(data1)
            # des2 = extract_kp(data2)
            des1 = extract_fingernet_minutiae(path1)
            des2 = extract_fingernet_minutiae(path2)
            if des1 is None or des2 is None:
                continue
            result = match(des1, des2)  
            print(str(result) + ' ' + str(issame))
            results.append((result, issame))
    print(len(results))
    results.sort(key=lambda x:x[0])
    draw_results(results)
    tp, fp = 0, 0
    tn, fn = len(results) / 2, len(results) / 2
    max_i = -1
    max_acc = tp + tn
    max_acc_tpr = tp / (tp + fn)
    for i in range(len(results)):
        if results[i][1]:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        if tp + tn > max_acc:
            max_i = i
            max_acc = tp + tn
            max_acc_tpr = tp / (tp + fn)
    if max_i == len(results) - 1:
        thresh = np.inf
    elif max_i == -1:
        thresh = -np.inf
    else:
        thresh = (results[max_i][0] + results[max_i + 1][0]) / 2
    print('--------------clean------------------')
    print(thresh)
    # TPR
    print(max_acc_tpr)
    with open('verification_result.txt', 'a') as f:
        f.write('fingernet: clean, thresh:{:.4f}, tpr:{:.4f}\n\n'.format(thresh, max_acc_tpr))
    return thresh
    
        
    # traditional method
    # for thresh in range(80, 90):
    #     count = 0
    #     for idx in range(len(add1)):
    #         for i in range(8):
    #             print(str(idx) + '-' + str(i))
    #             path1 = add1[idx][i]
    #             path2 = add2[idx][i]
    #             data1 = cv2.imread(add1[idx][i], cv2.IMREAD_GRAYSCALE)
    #             data2 = cv2.imread(add2[idx][i], cv2.IMREAD_GRAYSCALE)
    #             issame = issame_list[idx * 8 + i]
    #             des1 = extract_kp(data1)
    #             des2 = extract_kp(data2)
    #             # des1 = extract_fingernet_minutiae(path1)
    #             # des2 = extract_fingernet_minutiae(path2)
    #             if des1 is None or des2 is None:
    #                 continue
    #             result, _ = match(des1, des2, thresh)
    #             # print('**label = {}'.format(~issame))
    #             if result and not issame:  # match, True positive
    #                 count += 1
    #                 # print('True-True: {} / {}'.format(count, idx*8 + i + 1))
    #             # if not result and issame:  # not match
    #             #     count += 1
    #             #     print('False-False: {} / {}'.format(count, idx*8 + i + 1))
    #     print('--------------clean-{}------------------'.format(name))
    #     print(thresh)
    #     # TPR
    #     print(count * 2 / (len(issame_list) * 8))
    #     with open('verification_result.txt', 'a') as f:
    #         f.write('fingernet: clean-{}, thresh:{:d}, tpr:{:.4f}'.format(name, thresh, count * 2 / (len(issame_list) * 8)))

def verify_final(name, thresh = 0.8691402330375272):
    # verification: sample clean-adv配对
    root = './datapaths/datapath_MHS_clean_test.pkl'
    root_adv = './datapaths/datapath_MHS_{}_test.pkl'.format(name)

    valid_1, valid_2, issame_list, add1, add2 = get_valid_data(root, root_adv, num=68)
    issame_list = np.array(issame_list, dtype=bool)[:, 0]
    count = 0
    for idx in range(len(add1)):
        for i in range(8):
            print(str(idx) + '-' + str(i))
            path1 = add1[idx][i]
            path2 = add2[idx][i]
            data1 = cv2.imread(add1[idx][i], cv2.IMREAD_GRAYSCALE)
            data2 = cv2.imread(add2[idx][i], cv2.IMREAD_GRAYSCALE)
            issame = not issame_list[idx * 8 + i]
            # des1 = extract_kp(data1)
            # des2 = extract_kp(data2)
            des1 = extract_fingernet_minutiae(path1)
            des2 = extract_fingernet_minutiae(path2)
            if des1 is None or des2 is None:
                continue
            result = match(des1, des2)  
            print(str(result) + ' ' + str(issame))
            if result < thresh and issame:
                count += 1
    print('--------------clean-{}------------------'.format(name))
    print(thresh)
    # TPR
    print(count / (4 * len(add1)))
    with open('verification_result.txt', 'a') as f:
        f.write('fingernet: clean-{}, thresh:{:.4f}, tpr:{:.4f}\n\n'.format(name, thresh, count / (4 * len(add1))))

def verify_save(method, save_name='clean'):
    root = './datapaths/datapath_MHS_clean_test.pkl'
    root_adv = './datapaths/datapath_MHS_{}_test.pkl'.format(method)
    valid_1, valid_2, issame_list, add1, add2 = get_valid_data(root, root_adv, num=68)
    results = []
    for idx in range(len(add1)):
        for i in range(8):
            print(str(idx) + '-' + str(i))
            path1 = add1[idx][i]
            path2 = add2[idx][i]
            issame = not issame_list[idx * 8 + i]
            des1 = extract_fingernet_minutiae(path1)
            des2 = extract_fingernet_minutiae(path2)
            if des1 is None or des2 is None:
                continue
            result = match(des1, des2)  
            print(str(result) + ' ' + str(issame))
            results.append((result, issame))
    results.sort(key=lambda x:x[0])
    tpr, fpr, tnr, fnr = [], [], [], []
    tp, fp, tn, fn = 0, 0, len(results) / 2, len(results) / 2
    tpr.append(tp)
    fpr.append(fp)
    tnr.append(tn)
    fnr.append(fn)
    for i in range(len(results)):
        if results[i][1]:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        tpr.append(tp)
        fpr.append(fp)
        tnr.append(tn)
        fnr.append(fn)
    np.save('./roc_det/{}_tpr.npy'.format(save_name), np.array(tpr) / (len(results) / 2))
    np.save('./roc_det/{}_fpr.npy'.format(save_name), np.array(fpr) / (len(results) / 2))
    np.save('./roc_det/{}_tnr.npy'.format(save_name), np.array(tnr) / (len(results) / 2))
    np.save('./roc_det/{}_fnr.npy'.format(save_name), np.array(fnr) / (len(results) / 2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='clean')
    parser.add_argument('--save_name', type=str, default='clean')
    args = parser.parse_args()
    # Verification
    thresh = verify()
    verify_final(args.method, thresh=thresh)
    # Identification
    # identify_minutiae(args.method)
