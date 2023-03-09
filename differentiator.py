#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-21
# @Author  : Emily Wenger (ewenger@uchicago.edu)

import datetime
import time
import torch
import numpy as np
# import tensorflow as tf
# from fawkes.utils import preprocess, reverse_preprocess
# from keras.utils import Progbar
import pytorch_ssim
from L_orientation import ridge_orient
from cal_contrast import cal_contrast
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"



class FawkesMaskGeneration:
    # if the attack is trying to mimic a target image or a neuron vector
    MIMIC_IMG = True
    # number of iterations to perform gradient descent
    MAX_ITERATIONS = 10000
    # larger values converge faster to less accurate results
    LEARNING_RATE = 1e-2
    # the initial constant c to pick as a first guess
    INITIAL_CONST = 1
    # pixel intensity range
    INTENSITY_RANGE = 'imagenet'
    # threshold for distance
    L_THRESHOLD = 0.03
    # whether keep the final result or the best result
    KEEP_FINAL = False
    # max_val of image
    MAX_VAL = 1
    MAXIMIZE = False
    IMAGE_SHAPE = (224, 224, 3)
    RATIO = 1.0
    LIMIT_DIST = False
    LOSS_TYPE = 'features'  # use features (original Fawkes) or gradients (Witches Brew) to run Fawkes?

    def __init__(self, bottleneck_model_ls, mimic_img=MIMIC_IMG,
                 batch_size=6, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, initial_const=INITIAL_CONST,
                 intensity_range=INTENSITY_RANGE, l_threshold=L_THRESHOLD,
                 max_val=MAX_VAL, keep_final=KEEP_FINAL, maximize=MAXIMIZE, image_shape=IMAGE_SHAPE, verbose=1,
                 ratio=RATIO, limit_dist=LIMIT_DIST, loss_method=LOSS_TYPE, tanh_process=True,
                 save_last_on_failed=True, draw=False):

        assert intensity_range in {'raw', 'imagenet', 'inception', 'mnist', 'finger'}

        # constant used for tanh transformation to avoid corner cases

        self.it = 0
        self.tanh_constant = 2 - 1e-6
        self.save_last_on_failed = save_last_on_failed
        self.MIMIC_IMG = mimic_img
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.intensity_range = intensity_range
        self.l_threshold = l_threshold
        self.max_val = max_val
        self.keep_final = keep_final
        self.verbose = verbose
        self.maximize = maximize
        self.learning_rate = learning_rate
        self.ratio = ratio
        self.limit_dist = limit_dist
        self.single_shape = list(image_shape)
        self.bottleneck_models = bottleneck_model_ls
        self.loss_method = loss_method
        self.tanh_process = tanh_process
        self.draw = draw
        print('protecter th: {:02f}'.format(self.l_threshold))

    @staticmethod
    def resize_tensor(input_tensor, model_input_shape):
        if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
            return input_tensor
        resized_tensor = tf.image.resize(input_tensor, model_input_shape[:2])
        return resized_tensor

    def preprocess_arctanh(self, imgs):
        """ Do tan preprocess """
        # imgs = reverse_preprocess(imgs, self.intensity_range)
        imgs = imgs / 255.0
        imgs = imgs - 0.5
        imgs = imgs * self.tanh_constant
        tanh_imgs = np.arctanh(imgs)
        return tanh_imgs

    def reverse_arctanh(self, imgs):
        # raw_img = (tf.tanh(imgs) / self.tanh_constant + 0.5) * 255
        raw_img = (torch.tanh(imgs) / self.tanh_constant + 0.5) * 255
        return raw_img

    def input_space_process(self, img):
        if self.intensity_range == 'imagenet':
            mean = np.repeat([[[[103.939, 116.779, 123.68]]]], len(img), axis=0)
            raw_img = (img[..., ::-1] - mean)
        else:
            raw_img = img
        return raw_img

    def clipping(self, imgs):
        # imgs = reverse_preprocess(imgs, self.intensity_range)
        imgs = torch.clamp(imgs, 0, self.max_val)
        # imgs = preprocess(imgs, self.intensity_range)
        return imgs

    def calc_dissim(self, source_raw, source_mod_raw):
        # msssim_split = tf.image.ssim(source_raw, source_mod_raw, max_val=255.0)
        # dist_raw = (1.0 - tf.stack(msssim_split)) / 2.0
        # dist = tf.maximum(dist_raw - self.l_threshold, 0.0)
        # dist_raw_avg = tf.reduce_mean(dist_raw)
        # dist_sum = tf.reduce_sum(dist)
        msssim_split = torch.zeros((len(source_raw)))  # [6]
        for i in range(len(source_raw)):
            msssim_split[i] = pytorch_ssim.ssim(source_raw[i].unsqueeze(0).double(), source_mod_raw[i].unsqueeze(0).double())
        dist_raw = (1.0 - msssim_split.float().cuda()) / 2.0
        dist = torch.max(dist_raw - self.l_threshold, torch.tensor(0.0).cuda())
        dist_raw_avg = torch.mean(dist_raw)
        dist_sum = torch.sum(dist)

        return dist, dist_raw, dist_sum, dist_raw_avg

    def calc_bottlesim(self, source_raw, target_raw, original_raw):
        """ original Fawkes loss function. """
        bottlesim = 0.0
        bottlesim_sum = 0.0
        # make sure everything is the right size.
        model_input_shape = self.single_shape
        # cur_aimg_input = self.resize_tensor(source_raw, model_input_shape)
        cur_aimg_input = source_raw.float()
        if target_raw is not None:
            # cur_timg_input = self.resize_tensor(target_raw, model_input_shape)
            cur_timg_input = target_raw.float()
        for bottleneck_model in self.bottleneck_models:
            # if tape is not None:
            #     try:
            #         tape.watch(bottleneck_model.model.variables)
            #     except AttributeError:
            #         tape.watch(bottleneck_model.variables)
            # get the respective feature space reprs.
            bottleneck_a = bottleneck_model(cur_aimg_input.cuda())  # 6*1000
            if self.maximize:  # maximize=False
                bottleneck_s = bottleneck_model(original_raw.cuda())
                bottleneck_diff = bottleneck_a - bottleneck_s  # todo no need target image???
                scale_factor = torch.sqrt(torch.sum(torch.pow(bottleneck_s, 2), dim=1))
            else:  # todo with target
                # bottleneck_t = bottleneck_model(cur_timg_input)
                bottleneck_t = cur_timg_input  # .unsqueeze(0).repeat(6, 1)  # 6*1000
                bottleneck_diff = bottleneck_t - bottleneck_a
                scale_factor = torch.sqrt(torch.sum(torch.pow(bottleneck_t, 2), dim=1))
            # cur_bottlesim = tf.reduce_sum(tf.square(bottleneck_diff), axis=1)
            cur_bottlesim = torch.sum(torch.pow(bottleneck_diff, 2), dim=1)
            cur_bottlesim = cur_bottlesim / scale_factor
            bottlesim += cur_bottlesim
            # bottlesim_sum += tf.reduce_sum(cur_bottlesim)
            bottlesim_sum += torch.sum(cur_bottlesim)
        return bottlesim, bottlesim_sum

    def L_orientation_target(self, ridge, t_images, adv_images):
        # let orientation of adv_images close to target images
        distance = torch.zeros((len(t_images))).cuda()
        for i in range(len(t_images)):
            images_orient = ridge(t_images[i]).masked_fill(~mask.to('cuda'), value=0)
            adv_images_orient = ridge(adv_images[i]).masked_fill(~mask.to('cuda'), value=0)

            diff = torch.abs(adv_images_orient - images_orient)
            d = torch.sin(diff)
            d = d.abs()
            distance[i] = (torch.mean(d))
        return distance

    def L_contrast(self, contrast, images, adv_images):
        # pairwise distance, 1 if similiar, 0 if dissimiliar
        distance = torch.zeros((len(images))).cuda()
        for i in range(len(images)):
            local1, _, sal1 = contrast(images[i])
            local2, _, sal2 = contrast(adv_images[i])
            sal1, sal2 = sal1.unsqueeze(0).repeat(3, 1, 1), sal2.unsqueeze(0).repeat(3, 1, 1)
            con1 = torch.mul(local1, sal1)
            con2 = torch.mul(local2, sal2)
            distance[i] = torch.sum(torch.nn.functional.relu(con2 - con1))
        return distance

    def compute_feature_loss(self, aimg_raw, simg_raw, timg_raw, aimg_input, timg_emb, simg_input):
        """ Compute input space + feature space loss.
        """
        input_space_loss, dist_raw, input_space_loss_sum, input_space_loss_raw_avg = self.calc_dissim(aimg_raw,
                                                                                                      simg_raw)

        # orientation loss and contrast loss 没有使用，不是Fawkes的内容
        # ridge = ridge_orient()
        # contrast = cal_contrast()
        # loss_orientation = self.L_orientation_target(ridge, simg_raw, aimg_raw)  # [6]
        # loss_contrast = self.L_contrast(contrast, simg_raw, aimg_raw)  # [6]
        feature_space_loss, feature_space_loss_sum = self.calc_bottlesim(aimg_input, timg_emb, simg_input)

        if self.maximize:  # maximize=True
            # loss = self.const * tf.square(input_space_loss) - feature_space_loss * self.const_diff
            loss = self.const * torch.pow((input_space_loss), 2) - feature_space_loss * self.const_diff.cuda()
        else:  # maximize=False, with target
            if self.it <= self.MAX_ITERATIONS:
                loss = self.const * torch.pow((input_space_loss), 2) + 1 * feature_space_loss
                # loss = -feature_space_loss - 100 * loss_orientation + 0.01 * loss_contrast

        # loss_sum = tf.reduce_sum(loss)
        loss_sum = torch.sum(loss)
        return loss_sum, feature_space_loss, input_space_loss_raw_avg, dist_raw

    def compute(self, source_imgs, target_imgs=None, target_embs=None):
        """ Main function that runs cloak generation. """
        start_time = time.time()
        adv_imgs = []
        for idx in range(0, len(source_imgs), self.batch_size):  # self.batch-size=6
            print('processing image %d at %s' % (idx + 1, datetime.datetime.now()))
            adv_img = self.compute_batch(source_imgs[idx:idx + self.batch_size],
                                         target_imgs[idx:idx + self.batch_size] if target_imgs is not None else None,
                                         target_embs[idx:idx + self.batch_size] if target_embs is not None else None)
            adv_imgs.extend(adv_img)
        elapsed_time = time.time() - start_time
        print('protection cost %f s' % elapsed_time)
        return adv_imgs

    def compute_batch(self, source_imgs, target_imgs=None, target_embs=None, retry=True):  # images: N*3*224*224, embs: N*1000
        """ TF2 method to generate the cloak. """
        # preprocess images.
        global progressbar
        nb_imgs = source_imgs.shape[0]
        # print(nb_imgs)

        # make sure source/target images are an array
        source_imgs = np.array(source_imgs, dtype=np.float32)
        # if target_imgs is not None:
        #     target_imgs = np.array(target_imgs, dtype=np.float32)

        # metrics to test
        best_bottlesim = [0] * nb_imgs if self.maximize else [np.inf] * nb_imgs
        best_adv = torch.zeros(source_imgs.shape)

        # convert to tanh-space
        # simg_tanh = self.preprocess_arctanh(source_imgs)
        simg_tanh = source_imgs
        if target_imgs is not None:
            # timg_tanh = self.preprocess_arctanh(target_imgs)
            timg_tanh = target_imgs
        # self.modifier = tf.Variable(np.random.uniform(-1, 1, tuple([len(source_imgs)] + self.single_shape)) * 1e-4,
        #                             dtype=tf.float32)
        self.modifier = torch.FloatTensor(np.random.uniform(-1, 1, tuple([len(source_imgs)] + self.single_shape)) * 1e-4)
        self.modifier.requires_grad_(True)

        # make the optimizer
        # optimizer = tf.keras.optimizers.Adadelta(float(self.learning_rate))
        optimizer = torch.optim.Adam([self.modifier], lr=self.learning_rate)
        # const_numpy = np.ones(len(source_imgs)) * self.initial_const
        const_numpy = self.initial_const
        # self.const = tf.Variable(const_numpy, dtype=np.float32)
        self.const = torch.tensor(const_numpy)

        const_diff_numpy = np.ones(len(source_imgs)) * 1.0
        # self.const_diff = tf.Variable(const_diff_numpy, dtype=np.float32)
        self.const_diff = torch.tensor(const_diff_numpy)

        # get the modifier
        # if self.verbose == 0:
        #     progressbar = Progbar(
        #         self.MAX_ITERATIONS, width=30, verbose=1
        #     )
        # watch relevant variables.
        # simg_tanh = tf.Variable(simg_tanh, dtype=np.float32)
        # simg_raw = tf.Variable(source_imgs, dtype=np.float32)
        simg_tanh = torch.FloatTensor(simg_tanh)
        simg_raw = torch.FloatTensor(source_imgs)
        if target_imgs is not None:
            # timg_raw = tf.Variable(timg_tanh, dtype=np.float32)
            timg_raw = torch.tensor(timg_tanh)
        else:
            timg_raw = target_imgs
        # run the attack
        outside_list = np.ones(len(source_imgs))
        self.it = 0

        costs = []
        while self.it < self.MAX_ITERATIONS:

            self.it += 1
            # print('Before optimization:')
            # print(self.modifier)

            # Convert from tanh for DISSIM
            # aimg_raw = self.reverse_arctanh(simg_tanh + self.modifier)
            aimg_raw = simg_tanh + self.modifier

            actual_modifier = aimg_raw - simg_raw
            # actual_modifier = tf.clip_by_value(actual_modifier, -15.0, 15.0)
            actual_modifier = torch.clamp(actual_modifier, -8.0/255, 8.0/255)  # todo epsilon
            aimg_raw = simg_raw + actual_modifier

            # simg_raw = self.reverse_arctanh(simg_tanh)
            simg_raw = simg_tanh

            # Convert further preprocess for bottleneck
            aimg_input = self.input_space_process(aimg_raw)
            # if target_imgs is not None:
            #     timg_input = self.input_space_process(timg_raw)
            # else:
            #     timg_input = None
            simg_input = self.input_space_process(simg_raw)

            # get the feature space loss.
            loss, internal_dist, input_dist_avg, dist_raw = self.compute_feature_loss(
                aimg_raw, simg_raw, timg_raw, aimg_input, target_embs, simg_input)
            costs.append(loss.detach().cpu())
            # print(loss, internal_dist, input_dist_avg, dist_raw)

            # compute gradients
            # grad = tape.gradient(loss, [self.modifier])
            # optimizer.apply_gradients(zip(grad, [self.modifier]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if self.it == 1:
            #     # self.modifier = tf.Variable(self.modifier - tf.sign(grad[0]) * 0.01, dtype=tf.float32)
            #     self.modifier = torch.tensor(self.modifier - torch.sign(self.modifier.grad[0]) * 0.01)

            # print('After optimization:')
            # print(self.modifier)
            # for e, (input_dist, feature_d, mod_img) in enumerate((dist_raw, internal_dist, aimg_input)):
            for e in range(len(aimg_input)):
                if e >= nb_imgs:
                    break
                input_dist = dist_raw[e]
                feature_d = internal_dist[e]
                mod_img = aimg_input[e]
                input_dist = input_dist.cpu().detach().numpy()
                feature_d = feature_d.cpu().detach().numpy()

                if input_dist <= self.l_threshold * 0.9 and const_diff_numpy[e] <= 129:
                    const_diff_numpy[e] *= 2
                    if outside_list[e] == -1:
                        const_diff_numpy[e] = 1
                    outside_list[e] = 1
                elif input_dist >= self.l_threshold * 1.1 and const_diff_numpy[e] >= 1 / 129:
                    const_diff_numpy[e] /= 2

                    if outside_list[e] == 1:
                        const_diff_numpy[e] = 1
                    outside_list[e] = -1
                else:
                    const_diff_numpy[e] = 1.0
                    outside_list[e] = 0

                # if input_dist <= self.l_threshold * 1.1 and (
                if True and (
                        (feature_d < best_bottlesim[e] and (not self.maximize)) or (
                        feature_d > best_bottlesim[e] and self.maximize)):
                    best_bottlesim[e] = feature_d
                    best_adv[e] = mod_img.detach()
                    # print('update')

            # self.const_diff = tf.Variable(const_diff_numpy, dtype=np.float32)
            self.const_diff = torch.tensor(const_diff_numpy)

            # if self.verbose == 1:
            print("ITER {:0.2f}  Total Loss: {:.2f}; diff: {:.4f}".format(self.it, loss,
                                                                    np.mean(internal_dist.cpu().detach().numpy())))

            # if self.verbose == 0:
            #     progressbar.update(self.it)
        if self.draw:
            costs = np.array(costs)
            np.save('./convergence/fk.npy', costs)
            exit(0)
        if self.verbose == 1:
            print("Final diff: {:.4f}".format(np.mean(best_bottlesim)))
        print("\n")

        #if self.save_last_on_failed:
         #   for e, diff in enumerate(best_bottlesim):
          #      if diff < 0.3 and dist_raw[e] < 0.015 and internal_dist[e] > diff:
          #          best_adv[e] = aimg_input[e]

        best_adv = self.clipping(best_adv[:nb_imgs])
        return best_adv
