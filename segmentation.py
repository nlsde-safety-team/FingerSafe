import cv2
import numpy as np
import PIL.Image as Image


class Gray(object):

    def __call__(self, tensor):  # tensor: 3 * w * h
        # TODO: make efficient
        _, w, h = tensor.shape
        R = tensor[0]
        G = tensor[1]
        B = tensor[2]
        tmp = 0.299 * R + 0.587 * G + 0.114 * B
        tensor = tmp
        tensor = tensor.view(1, w, h)
        return tensor


# ---- START FUNCTIONS ----#

# display an image plus label and wait for key press to continue
def display_image(image, name):
    window_name = name
    # img_temp = cv2.resize(image, (2000, 1000))
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_mask(mask):
    assert len(mask.shape) == 3
    mask = mask[..., 0]
    width, height = mask.shape
    for width_left in range(width):
        pattern = np.zeros((height, 1)).squeeze()

        if np.sum(mask[width_left, ...] - pattern) != 0:
            break

    for width_right in range(width):
        pattern = np.zeros((height, 1)).squeeze()
        if np.sum(mask[width - width_right - 1, ...] - pattern) != 0:
            break

    for height_top in range(height):
        pattern = np.zeros((width, 1)).squeeze()
        if np.sum(mask[..., height_top] - pattern) != 0:
            break

    for height_bottom in range(width):
        pattern = np.zeros((width, 1)).squeeze()
        if np.sum(mask[..., height - height_bottom - 1] - pattern) != 0:
            break

    return width_left, width - width_right - 1, height_top, height - height_bottom - 1


def get_rotation(mask):
    assert len(mask.shape) == 3
    mask_temp = mask
    # display_image(mask, 'msk')
    mask = (mask * 255).astype('uint8')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    width, height = mask.shape
    # print(width, height)

    # p1: (middle width, 1/3 * height)
    # p2: (middle width, 2/3 * height)
    p1_width = int(width / 2)
    p1_height = int(height * 3 / 7)

    p2_width = int(width / 2)
    p2_height = int(height * 4 / 7)

    find = 0
    for p1_x in range(p1_width):
        # cv2.circle(mask_temp, (p1_width - p1_x, p1_height), 10, (0, 255, 0), 10)
        if mask[p1_width - p1_x, p1_height] == 0:
            find = 1
            break

    if find == 0:
        for p1_x in range(width - p1_width):
            if mask[p1_width + p1_x, p1_height] == 0:
                find = 1
                break
    if find == 0:
        raise ValueError('area not found')

    find = 0
    for p2_x in range(p2_width):
        # cv2.circle(mask_temp, (p2_width - p2_x, p2_height), 10, (0, 255, 0), 10)
        if mask[p2_width - p2_x, p2_height] == 0:
            find = 1
            break

    if find == 0:
        for p2_x in range(width - p2_width):
            if mask[p2_width + p2_x, p2_height] == 0:
                find = 1
                break
    if find == 0:
        raise ValueError('area not found')
    # cv2.circle(mask_temp, (p1_x, p1_height), 10, (0, 0, 255), 10)
    # cv2.circle(mask_temp, (p2_x, p2_height), 10, (0, 0, 255), 10)
    # display_image(mask_temp, 'msk')
    delta_y = p2_height - p1_height
    delta_x = p2_x - p1_x
    theta = np.arctan2(delta_y, delta_x)
    return theta
import math
def rotate(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    # sin = math.sin(radians)
    # cos = math.cos(radians)
    # bound_w = int((height * abs(sin)) + (width * abs(cos)))
    # bound_h = int((height * abs(cos)) + (width * abs(sin)))

    # rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    # rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    # rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    cos = np.abs(rotation_mat[0, 0])
    sin = np.abs(rotation_mat[0, 1])

    new_w = height * sin + width * cos
    new_h = height * cos + width * sin

    rotation_mat[0, 2] += new_w * 0.5 - image_center[0]
    rotation_mat[1, 2] += new_h * 0.5 - image_center[1]

    w = int(np.round(new_w))
    h = int(np.round(new_h))

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (w, h))
    return rotated_mat


def get_truncate(img, mask):
    # all things here are empirical. don't surprise if it don't work
    # if it don't work, simply divide by 2
    assert len(img.shape) == 3
    width, height, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    central = img_gray[int(width / 2), ...]
    up = img_gray[int(width / 2) + 50, ...]
    down = img_gray[int(width / 2) - 50, ...]
    fused = (central + up + down) / 3
    fused = fused[200: -50]
    fused = 2 * (fused - np.min(fused)) / (np.max(fused) - np.min(fused)) - 1
    index = np.argmin(fused)
    return index + 200


def segment(img_BGR, trun):
    # ---- MAIN ----#
    # read in image into openCV BGR and grayscale
    # image_path = "./segment/single.jpg"
    #
    # img_BGR = cv2.imread(image_path, 3)

    # img_BGR = cv2.resize(img_BGR, (2000, 1000))
    # display_image(img_BGR, "BGR")

    img_grayscale = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    img_YCC = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCR_CB)
    img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)

    Cr = img_YCC[..., 2]
    H = img_HSV[..., 0]


    cv2.imwrite('./segment/test/Cr.jpg', Cr)
    cv2.imwrite('./segment/test/H.jpg', H)

    # convert to CMYK
    # Extract channels
    img_BGR = img_BGR.astype('float') / 255.
    with np.errstate(invalid='ignore', divide='ignore'):
        K = 1 - np.max(img_BGR, axis=2)
        C = (1 - img_BGR[..., 2] - K) / (1 - K)
        M = (1 - img_BGR[..., 1] - K) / (1 - K)
        Y = (1 - img_BGR[..., 0] - K) / (1 - K)
    M = (M * 255).astype('uint8')

    threshold_value_Cr, threshold_image_Cr = cv2.threshold(Cr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold_value_H, threshold_image_H = cv2.threshold(H, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imwrite('./segment/test/thres_Cr.jpg', threshold_image_Cr)
    cv2.imwrite('./segment/test/thres_H.jpg', threshold_image_H)

    mask_H_binary = threshold_image_H / 255
    mask_Cr_binary = threshold_image_Cr / 255
    threshold_image_binary = cv2.bitwise_and(mask_H_binary, mask_Cr_binary)
    threshold_image = (threshold_image_binary * 255).astype('uint8')

    cv2.imwrite('./segment/test/thres.jpg', threshold_image)

    # kernel_size = 31  
    kernel_size = 31  
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # dilation and erode to eliminate spurious features
    threshold_image = cv2.erode(threshold_image, kernel, iterations=1)
    threshold_image = cv2.dilate(threshold_image, kernel, iterations=1)
    cv2.imwrite('./segment/test/thres_erode.jpg', threshold_image)

    # find contours and return the largest
    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(threshold_image, [contours[k]], 0)

    cv2.imwrite('./segment/test/thres_contours.jpg', threshold_image)

    threshold_image_binary = threshold_image / 255
    threshold_image_binary = np.repeat(threshold_image_binary[:, :, np.newaxis], 3, axis=2)
    img_face_only = np.multiply(threshold_image_binary, img_BGR)

    # mask select the image so now only fingerprint area is retained
    w1, w2, h1, h2 = get_mask(threshold_image_binary)
    # fin_img = img_face_only[w1: w2, h1: h2] 
    fin_img = img_BGR[w1: w2, h1: h2] 
    fin_mask = threshold_image_binary[w1: w2, h1: h2]
    position = (w1, w2, h1, h2)

    cv2.imwrite('./segment/test/before_rotate_mask.jpg', fin_mask * 255)

    theta = get_rotation(fin_mask)
    fin_img = rotate(fin_img, (theta * 180) / np.pi - 90)
    fin_mask = rotate(fin_mask, (theta * 180) / np.pi - 90)
    fin_img = rotate(fin_img, (theta * 180) / np.pi)
    fin_mask = rotate(fin_mask, (theta * 180) / np.pi)
    fin_img = np.array(fin_img * 255).astype('uint8')
    fin_mask = np.array(fin_mask * 255).astype('uint8')
    cv2.imwrite('./segment/test/rotate_mask.jpg', fin_mask)

    # cut the rotated image
    w1, w2, h1, h2 = get_mask(fin_mask)
    fin_img = fin_img  # [w1: w2, h1: h2]  # BGR, w*h*
    fin_mask = fin_mask  # [w1: w2, h1: h2]

    fin_img = fin_img * (fin_mask / 255)

    # display_image(fin_img, 'fi')
    # display_image(fin_mask, 'fm')

    # not quite robust
    # trun_index = get_truncate(fin_img, fin_mask)
    fin_height, fin_width, _ = fin_img.shape
    trun_index = int(fin_height / trun)  # empirically determined

    final_img = fin_img[:trun_index, :, :]
    final_mask = fin_mask[:trun_index, :, :]
    # final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)  # todo BGR2RGB
    final_mask = final_mask[:, :, 0]
    # return final_img, final_mask, (position, trun_index), img_BGR  # todo return

    print(position)
    cv2.imwrite('./segment/test/raw.jpg', img_BGR*255)
    cv2.imwrite('./segment/test/mask.jpg', final_img)
    cv2.imwrite('./segment/test/left_img.jpg', final_mask)
    # display_image(threshold_image, "thresholded image")
    # display_image(img_face_only, "segmented BGR")
    from process_dataset import IJCB2015
    ijcb = IJCB2015(cv2.cvtColor(final_img.astype(np.uint8), cv2.COLOR_BGR2GRAY))
    cv2.imwrite('./segment/test/MHS.jpg', ijcb)


def reverse_segment(fin_image, raw_pic, clean_pic, pos):
    ((w1, w2, h1, h2), thun) = pos
    print(pos)
    print(clean_pic.shape)
    print(raw_pic.shape)
    fin_image = cv2.resize(fin_image, (clean_pic.shape[1], clean_pic.shape[0]))
    noise = fin_image - clean_pic
    raw_pic[w1: w1+clean_pic.shape[0], h1: h1+clean_pic.shape[1], :] += noise
    return raw_pic, noise


if __name__ == '__main__':
    image_path = "./segment/test2.JPG"
    img_BGR = cv2.imread(image_path, 3)
    segment(img_BGR, trun=1.35)