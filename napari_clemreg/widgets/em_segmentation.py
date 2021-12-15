#!/usr/bin/env python3
# coding: utf-8
from magicgui import magic_factory
# from napari.layers import Labels, Image
# import numpy as np

# from dataloader import SparseDataGenerator
# from sparse_unet.model import SparseUnet

from skimage import transform
from skimage import exposure
from skimage.exposure import equalize_hist

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import ReLU
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Reshape
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import Callback
from keras.models import Model

from .sparse_unet.dataloader import SparseDataGenerator
from .sparse_unet.utils import SampleImageCallback
from .sparse_unet.utils import weighted_categorical_crossentropy
from .sparse_unet.utils import dice_coefficient

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
import math
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

        for batch_idx in range(self.batch_size):
            img_crop, mask_crop = self._random_crop(self.source[curr_idx],
                                                    self.target[curr_idx],
                                                    self.crop_shape)
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

        # return (w_0, w_1, 0)
        return (1, 3, 0)

    def _random_crop(self, img, mask, crop_shape):
        width, height = crop_shape
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)

        img_crop = img[y:y+height, x:x+width]
        mask_crop = mask[y:y+height, x:x+width]

        return img_crop, mask_crop

    def _z_score(self, input):
        return (input - np.mean(input)) / np.std(input)

class SparseUnet:
    def __init__(self, shape=(256,256,1)):

        self.shape = shape

        input_img = Input(self.shape, name='img')

        self.model = self.unet_2D(input_img)

    def down_block_2D(self, input_tensor, filters):

        x = Conv2D(filters=filters, kernel_size=(3,3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def up_block_2D(self, input_tensor, concat_layer, filters):

        x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2))(input_tensor)

        x = Concatenate()([x, concat_layer])

        x = Conv2D(filters=filters, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def unet_2D(self, input_tensor, filters=32):

        d1 = self.down_block_2D(input_tensor, filters=filters)
        p1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(d1)
        d2 = self.down_block_2D(p1, filters=filters*2)
        p2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(d2)
        d3 = self.down_block_2D(p2, filters=filters*4)
        p3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(d3)

        d4 = self.down_block_2D(p3, filters=filters*8)

        u1 = self.up_block_2D(d4, d3, filters=filters*4)
        u2 = self.up_block_2D(u1, d2, filters=filters*2)
        u3 = self.up_block_2D(u2, d1, filters=filters)

        # Returns one-hot-encoded semantic segmentation mask where 0 is bakcground, 1 is mito and 2 is None (weight zero)
        output_tensor = Conv2D(filters=3, kernel_size=(1,1), activation='softmax')(u3)

        return Model(inputs=[input_tensor], outputs=[output_tensor])

    def train(self,
              src,
              tgt,
              out_dir,
              epochs=100,
              batch_size=32,
              dense=False,
              log_name='log.csv',
              model_name='sparse_unet'):

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(os.path.join(out_dir, 'ckpt')):
            os.makedirs(os.path.join(out_dir, 'ckpt'))

        train_generator = SparseDataGenerator(src,
                                              tgt,
                                              batch_size=batch_size,
                                              crop_shape=self.shape[:2])

        val_generator = SparseDataGenerator(src,
                                            tgt,
                                            batch_size=batch_size,
                                            crop_shape=self.shape[:2],
                                            augment=False)

        sample_batch = val_generator[0][0]
        sample_img = SampleImageCallback(self.model,
                                         sample_batch,
                                         out_dir,
                                         save=True)

        weight_zero, weight_one, weight_two = train_generator.batch_weights()

        self.model.compile(optimizer='adam',
                           loss=weighted_categorical_crossentropy(np.array([weight_zero, weight_one, weight_two])),
                           metrics=[dice_coefficient])

        self.model.summary()

        # self.model.load_weights(model_name)

        csv_logger = CSVLogger(os.path.join(out_dir, log_name))

        ckpt_name =  'ckpt/' + model_name + '_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.hdf5'

        model_ckpt = ModelCheckpoint(os.path.join(out_dir, ckpt_name),
                                     verbose=1,
                                     save_best_only=False,
                                     save_weights_only=True)

        self.model.fit_generator(generator=train_generator,
                                 validation_data=val_generator,
                                 validation_steps=math.floor(len(val_generator))/batch_size,
                                 epochs=epochs,
                                 shuffle=False,
                                 callbacks=[csv_logger,
                                            model_ckpt,
                                            sample_img])

    def predict(self, input, tile_shape):

        pad_in = np.zeros((math.ceil(input.shape[0] / tile_shape[0]) * tile_shape[0], math.ceil(input.shape[1] / tile_shape[1]) * tile_shape[1]))

        pad_in[:input.shape[0],:input.shape[1]] = input

        pad_out = np.zeros(pad_in.shape)

        for x in range(math.ceil(input.shape[0] / tile_shape[0])):
            for y in range(math.ceil(input.shape[1] / tile_shape[1])):
                x_a = x * tile_shape[0]
                x_b = x * tile_shape[0] + tile_shape[0]

                y_a = y * tile_shape[1]
                y_b = y * tile_shape[1] + tile_shape[1]
                pad_out[x_a:x_b,y_a:y_b] = self.tile_predict(pad_in[x_a:x_b,y_a:y_b])

        return pad_out[:input.shape[0],:input.shape[1]]

    def tile_predict(self, input):

        exp_input = np.zeros((1, self.shape[0], self.shape[1], 1))

        exp_input[0,:,:,0] = exposure.equalize_hist(transform.resize(input, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True))

        return transform.resize(np.squeeze(self.model.predict(exp_input))[:,:,1], input.shape, mode='constant', preserve_range=True)

    def save(self, model_name):
        self.model.save_weights(model_name)

    def load(self, model_name):
        self.model.load_weights(model_name)


@magic_factory
def train_model(source: Image,
                target: Labels): # -> Image:

    # train_generator = SparseDataGenerator(source, target, crop_shape=(512,512))
    # print('Len', len(train_generator))

    model = SparseUnet(shape=(512,512,1))

    model.train(source,
                target,
                out_dir='output_dir',
                epochs=10,
                batch_size=16)

    model.predict(source.data, tile_shape=(512,512))

    # return Image(cropped_images)
