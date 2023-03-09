#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/

import argparse
import glob
import logging
import os
import sys
from FingerprintDataset import FingerprintTrain, Fingerprint_Mask, FingerprintTest
from torch.utils.data import DataLoader
import random
from InjectNoise import find_target



# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# os.environ["KMP_AFFINITY"] = "noverbose"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
#
# tf.get_logger().setLevel('ERROR')
# tf.autograph.set_verbosity(3)

import numpy as np
from models.inception_resnet_v1 import ResNet as ResNet
import torch
from differentiator import FawkesMaskGeneration
from utils import init_gpu, dump_image, reverse_process_cloaked, \
    Faces, filter_image_paths

# from fawkes.align_face import aligner
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def generate_cloak_images(protector, image_X, target_img=None, target_emb=None):
    cloaked_image_X = protector.compute(image_X, target_img, target_emb)
    return cloaked_image_X


def load_extractor(weight_name):
    print(weight_name)

    resnet = ResNet(
        classify=False,
        nclasses=268
    ).to('cuda')
    resnet.load_state_dict(torch.load(weight_name))
    # print(resnet.classify)
    resnet.eval()
    return resnet

IMG_SIZE = 224
PREPROCESS = 'raw'


class Fawkes(object):
    def __init__(self, feature_extractor, gpu, batch_size, mode="finger"):

        self.feature_extractor = feature_extractor
        self.gpu = gpu
        self.batch_size = batch_size
        self.mode = mode
        th, lr, extractors = self.mode2param(self.mode)
        self.th = th
        self.lr = lr
        # if gpu is not None:
        #     init_gpu(gpu)

        # self.aligner = aligner()

        self.protector = None
        self.protector_param = None
        self.feature_extractors_ls = [load_extractor(name) for name in extractors]

    def mode2param(self, mode):
        if mode == 'low':
            th = 0.004
            max_step = 40
            lr = 25
            extractors = ["extractor_2"]

        elif mode == 'mid':
            th = 0.012
            max_step = 75
            lr = 20
            extractors = ["extractor_0", "extractor_2"]

        elif mode == 'high':
            th = 0.017
            max_step = 150
            lr = 15
            extractors = ["extractor_0", "extractor_2"]
        elif mode == 'finger':
            th = 0.1
            lr = 0.01
            extractors = ["./best_models/clean_split_1009.pth"]

        else:
            raise Exception("mode must be one of 'min', 'low', 'mid', 'high'")
        return th, lr, extractors

    def run_protection(self, image_paths, th=0.04, sd=1e4, lr=10, max_step=500, batch_size=1, format='bmp',
                       separate_target=True, debug=False, no_align=False, exp="", maximize=False,
                       save_last_on_failed=True, output='sample_fk_th_5e-1', draw = False):

        current_param = "-".join([str(x) for x in [self.th, sd, self.lr, max_step, batch_size, format,
                                                   separate_target, debug]])
        # define protector
        self.protector = FawkesMaskGeneration(self.feature_extractors_ls,
                                              batch_size=batch_size,
                                              mimic_img=True,
                                              intensity_range=PREPROCESS,
                                              initial_const=sd,
                                              learning_rate=self.lr,
                                              max_iterations=max_step,
                                              l_threshold=th,
                                              verbose=debug,
                                              maximize=maximize,
                                              keep_final=False,
                                              image_shape=(3, 224, 224),
                                              loss_method='features',
                                              tanh_process=True,
                                              save_last_on_failed=save_last_on_failed,
                                              draw=draw
                                              )

        # image_paths, loaded_images = filter_image_paths(image_paths)

        if not image_paths:
            print("No images in the directory")
            return 3
        # todo load raw data
        dataroot = 'datasets/final/veri_test'
        source_set = FingerprintTest(dataroot)
        source_loader = DataLoader(dataset=source_set,
                                 batch_size=batch_size,
                                 shuffle=False)
        
        target_set = FingerprintTrain('datasets/final/train')


        for i_batch, (x, y, path) in enumerate(source_loader):
            original_images = x
            paths = path
            # target_emb = []
            # find target for each image
            # for i_image in range(len(x)):
            #     image = x[i_image]
            #     random_class = [random.randint(0, 267) for _ in range(8)]  # choose 8 targets from trainset
            #     print('target class: ' + str(random_class))
            #     target_set_random = [target_set.x_data[i * 6: (i + 1) * 6] for i in random_class]
            #
            #     target_emb_centers = []
            #     target_embs = []
            #     for iclass in target_set_random:
            #         target_images = torch.stack(iclass, 0)
            #         target_embedding = self.feature_extractors_ls[0](target_images.cuda())   # 6*1000
            #         target_embedding_center = torch.mean(target_embedding, dim=0)  # 1000
            #         target_emb_centers.append(target_embedding_center)
            #         target_embs.append(target_embedding)
            #
            #     # find the most dissimilar target class for source image
            #     source_embedding = self.feature_extractors_ls[0](original_images.cuda())  # 6*1000
            #     distance = 0
            #     target_index = 0
            #     for index in range(len(target_emb_centers)):
            #         t = target_emb_centers[index]  # 1*1000
            #         t = t.repeat(6, 1)  # 6*1000
            #         i_distance = torch.norm((t - source_embedding), p=2)
            #         if i_distance > distance:  # find the most dissimilar
            #             distance = i_distance
            #             target_index = index
            # # randomly pick an image from class T
            # # target_emb = target_embedding[random.randint(0, 5)].detach()
            #     target_emb.append(target_embs[target_index][random.randint(0, 5)].cpu().detach().numpy())  # 1*1000
            # target_emb = torch.from_numpy(np.array(target_emb)).cuda()  # 6*1000

            target_emb, target_images = find_target(original_images, target_set, model=self.feature_extractors_ls[0])
            original_images = np.array(original_images)

            if current_param != self.protector_param:
                self.protector_param = current_param
                if batch_size == -1:
                    batch_size = len(original_images)
            # todo here to generate
            protected_images = generate_cloak_images(self.protector, original_images,
                                                     target_img=target_images, target_emb=target_emb)
            export(paths, protected_images, output)
        print("Done!")
        return 1


import torchvision
def export(paths, images, output):
    unloader = torchvision.transforms.ToPILImage()
    for idx in range(len(images)):
        p, f = os.path.split(paths[0][idx])
        if p.find('train') != -1:
            new_path = p.replace('train', 'perturb_fk_e16')
        else:
            new_path = p.replace('test', output)

        os.makedirs(os.path.dirname(os.path.join(new_path, f)), exist_ok=True)
        img = unloader(images[idx].cpu().detach().squeeze(0))
        img.save(os.path.join(new_path, f))
        print('writing to: ' + os.path.join(new_path, f))

def main(*argv):
    if not argv:
        argv = list(sys.argv)

    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception as e:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', type=str,
                        help='the directory that contains images to run protection', default='datasets/final/veri_test/')
    parser.add_argument('--gpu', '-g', type=str,
                        help='the GPU id when using GPU for optimization', default='1')
    parser.add_argument('--mode', '-m', type=str,
                        help='cloak generation mode, select from min, low, mid, high. The higher the mode is, '
                             'the more perturbation added and stronger protection',
                        default='finger')
    parser.add_argument('--feature-extractor', type=str,
                        help="name of the feature extractor used for optimization",
                        default="arcface_extractor_0")
    parser.add_argument('--th', help='only relevant with mode=custom, DSSIM threshold for perturbation', type=float,
                        default=0.1)
    parser.add_argument('--max-step', help='only relevant with mode=custom, number of steps for optimization', type=int,
                        default=10)
    parser.add_argument('--sd', type=int, help='only relevant with mode=custom, penalty number, read more in the paper',
                        default=1e3)
    parser.add_argument('--lr', type=float, help='only relevant with mode=custom, learning rate', default=0.5)
    parser.add_argument('--batch-size', help="number of images to run optimization together", type=int, default=6)
    parser.add_argument('--separate_target', help="whether select separate targets for each faces in the directory",
                        action='store_true')
    parser.add_argument('--no-align', help="whether to detect and crop faces",
                        action='store_true')
    parser.add_argument('--debug', help="turn on debug and copy/paste the stdout when reporting an issue on github",
                        action='store_true')
    parser.add_argument('--format', type=str,
                        help="format of the output image",
                        default="bmp")
    parser.add_argument('--output', type=str, default='sample_fk_th_5e-1')

    args = parser.parse_args(argv[1:])
    args.no_align = True
    draw = False

    assert args.format in ['png', 'jpg', 'jpeg', 'bmp']
    if args.format == 'jpg':
        args.format = 'jpeg'

    image_paths = glob.glob(os.path.join(args.directory, "*"))
    image_paths = [path for path in image_paths if "_cloaked" not in path.split("/")[-1]]

    protector = Fawkes(args.feature_extractor, args.gpu, args.batch_size, mode=args.mode)

    protector.run_protection(image_paths, th=args.th, sd=args.sd, lr=args.lr,
                             max_step=args.max_step,
                             batch_size=args.batch_size, format=args.format,
                             separate_target=args.separate_target, debug=args.debug, no_align=args.no_align, output=args.output, draw=draw)


if __name__ == '__main__':
    main(*sys.argv)
    # todo other parameter settings
    # line 136：load the dataset
    # line 96: set threshold, lr
    # set epsilon：differentiator.py line 302
