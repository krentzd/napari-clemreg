#!/usr/bin/env python3
# coding: utf-8

import os
import random
import h5py
import glob
import imageio

import numpy as np
import tensorflow as tf
# import imgaug.augmenters as iaa

from skimage import transform
from skimage import exposure
from skimage import color
from skimage.io import imread

from keras import backend as K
from keras.utils import Sequence

from napari.layers import Labels, Image

# TODO: Implement DataGenerator that uses label file as training data and automatically crops it

class SparseDataGenerator(Sequence):

    def __init__(self,
                 source_layer: Image,
                 target_layer: Labels,
                 batch_size=16,
                 crop_shape=(256,256),
                 augment=True,
                 is_val=False):

        self.source = source_layer.data
        self.target = target_layer.data
        self.crop_shape = crop_shape
        self.batch_size = batch_size

        assert self.source.shape == self.target.shape, 'Source and target layers must have same shape!'
        assert crop_shape[0] < self.source.shape[1] and crop_shape[1] < self.source.shape[2], 'Crop must be smaller than training images'

        self.layer_idx = [i for (i, l) in enumerate(self.target) if np.sum(l) > 0]

        # self.seq = iaa.Sequential([
        #            iaa.Fliplr(0.5), # horizontal flips
        #            iaa.Flipud(0.5), # vertical flips
        #            iaa.Crop(percent=(0, 0.1)), # random crops
        #            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        #            iaa.LinearContrast((0.75, 1.5)),
        #            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
        #            iaa.Multiply((0.8, 1.2)),
        #            iaa.Affine(
        #                 scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #                 translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #                 rotate=(-25, 25),
        #                 shear=(-8, 8))
        #            ], random_order=True) # apply augmenters in random order

        # Initialise class
        # Identify layers with annotations

    def __getitem__(self, idx):
        img_batch = np.zeros((self.batch_size,
                              self.crop_shape[0],
                              self.crop_shape[1],
                              1))
        mask_batch = np.zeros((self.batch_size,
                              self.crop_shape[0],
                              self.crop_shape[1],
                              3))

        curr_idx = self.layer_idx[idx % len(self.layer_idx)]
        print(curr_idx)

        for batch_idx in range(self.batch_size):
            print(batch_idx)
            img_crop, mask_crop = self._random_crop(self.source[curr_idx],
                                                    self.target[curr_idx],
                                                    self.crop_shape)
            print(img_crop.shape, mask_crop.shape)
        # # Apply data augmentation
        # # Affine
        # # Gamma augmentation
        #
            img_batch[batch_idx,:,:,0] = np.clip(self._z_score(img_crop), -1, 1)
            mask_batch[batch_idx,:,:,0] = mask_crop == 1
            mask_batch[batch_idx,:,:,1] = mask_crop == 2
            mask_batch[batch_idx,:,:,2] = mask_crop == 0
        # print('got item')
        return np.stack(img_batch), mask_batch
        # Augment data
        #Return batch

    def __len__(self):
        # __, x_full_ y_full = self.target.shape
        # x_crop, y_crop = self.crop_shape
        return 1
        # return len(self.layer_idx) * (x_full / x_crop) * (y_full / y_crop)
        # Length of datagenerator is (1 / aug_freq) * num_of_annotated_layers * (x_full / x_crop) * (y_full / y_crop)

    def batch_weights(self):
        n_0 = np.sum(self.target == 1)
        n_1 = np.sum(self.target == 2)

        w_0 = n_1 / (n_0 + n_1)
        w_1 = n_0 / (n_0 + n_1)

        return (w_0, w_1, 0)

    def _random_crop(self, img, mask, crop_shape):
        print(img.shape, mask.shape)
        width, height = crop_shape
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        print('Random crop')
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)

        img_crop = img[y:y+height, x:x+width]
        mask_crop = mask[y:y+height, x:x+width]

        return img_crop, mask_crop

    def _z_score(self, input):
        return (input - np.mean(input)) / np.std(input)
