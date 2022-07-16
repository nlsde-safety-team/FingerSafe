import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator
from FingerprintDataset import Fingerprint_Seg
from sklearn.metrics import accuracy_score, precision_score, f1_score
import fingerprint_feature_extractor
from verification import get_valid_data


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
    _, des = orb.compute(image, keypoints)
    return des

# Returns feature descriptors for all images from the dataset
def get_feature_descriptors(dataset):
    feature_descriptors = {}
    for image_id, (image, image_label) in enumerate(dataset):
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
            matches = sorted(matches, key=lambda x: x.distance, reverse=False)  # sort matches based on feature distance

            best_matches = [m.distance for m in matches if m.distance < distance_threshold]
            best_matches_dict[train_image_id] = len(
                best_matches)  # matching function = length of best matches to given threshold
    best_matches_dict = sorted(best_matches_dict.items(), key=operator.itemgetter(1),
                               reverse=True)  # sort by value - feature distance
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


def match(des1, des2, score_threshold):
    # Matching between descriptors
    # Brute force match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)

    # Calculate score
    score = 0
    for match in matches:
        score += match.distance
    #print(score)
    #print(len(matches))
    if len(matches) == 0:
        avg = np.inf
    else:
        avg = score / len(matches)
    if avg < score_threshold:
        #print("Fingerprint matches.")
        return True, avg
    else:
        # print("Fingerprint does not match.")
        return False, avg


# Definition of identification scenario
def perform_identification_scenario(train_set, train_feature_descriptors, test_set, dist_threshold, rank):
    true_y = []
    pred_y = []
    total_prob = 0
    print("----- START, threshold = {}, rank = {} -----".format(dist_threshold, rank))
    index = 0
    for (test_image, test_label) in test_set:
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
        true_y.append(true_class)  # true_class
        pred_y.append(predicted_class)

    avg_probability = total_prob / len(test_set)
    print("Averaged probability for rank %d and threshold %d is %f " % (rank, dist_threshold, avg_probability))
    print("Accuracy for rank %d and threshold %d is %f " % (rank, dist_threshold, accuracy_score(true_y, pred_y)))
    return avg_probability

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


def verify(name):
    # verification: sample clean-adv pairing
    # extract minutiae, then use BFMatcher to check how good they are paired, then determine if those two images match.
    root = './datapaths/datapath_MHS_clean_test.pkl'
    root_adv = './datapaths/datapath_MHS_{}_test.pkl'.format(name)

    valid_1, valid_2, issame_list, add1, add2 = get_valid_data(root, root_adv, num=68)
    issame_list = np.array(issame_list, dtype=bool)[:, 0]
    print(issame_list)
    for thresh in range(80, 90):
        count = 0
        for idx in range(len(add1)):
            for i in range(8):
                data1 = cv2.imread(add1[idx][i], cv2.IMREAD_GRAYSCALE)
                data2 = cv2.imread(add2[idx][i], cv2.IMREAD_GRAYSCALE)
                issame = issame_list[idx * 8 + i]
                des1 = extract_kp(data1)
                des2 = extract_kp(data2)
                if des1 is None or des2 is None:
                    continue
                result, _ = match(des1, des2, thresh)
                if result and not issame:  # match, True positive
                    count += 1
        print('--------------clean-{}------------------'.format(name))
        print(thresh)
        # TPR
        print(count * 2 / (len(issame_list) * 8))


if __name__ == '__main__':
    verify('clean')
