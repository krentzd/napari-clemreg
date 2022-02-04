#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import csv
import h5py
import math

import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa

from skimage import transform
from skimage import exposure
from skimage.exposure import equalize_hist

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

from .dataloader import SparseNapariDataGenerator
from .utils import SampleImageCallback
from .utils import weighted_categorical_crossentropy
from .utils import dice_coefficient

class SparseUnet:
    def __init__(self, shape=(256,256,1)):

        self.shape = shape

        input_img = Input(self.shape, name='img')

        self.model = self.unet_2D(input_img)

    def down_block_2D(self, input_tensor, filters):

        x = Conv2D(filters=filters, kernel_size=(3,3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Dropout(0.5)(x)

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

        x = Dropout(0.5)(x)

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
              source,
              target,
              out_dir,
              epochs=100,
              batch_size=32,
              dense=False,
              log_name='log.csv',
              model_name='sparse_unet'):

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(os.path.join(out_dir, 'ckpt')):
            os.makedirs(os.path.join(out_dir, 'ckpt'))

        train_generator = SparseNapariDataGenerator(source,
                                                  target,
                                                  batch_size=batch_size,
                                                  shape=self.shape,
                                                  dense=dense,
                                                  augment=True)

        val_generator = SparseNapariDataGenerator(source,
                                                target,
                                                batch_size=batch_size,
                                                shape=self.shape,
                                                dense=dense,
                                                augment=False,
                                                is_val=True)

        sample_batch = val_generator[0][0]
        sample_img = SampleImageCallback(self.model,
                                         sample_batch,
                                         out_dir,
                                         save=True)

        weight_zero, weight_one, weight_two = train_generator.batch_weights()

        optim = Adam(learning_rate=0.001)
        # from: https://arxiv.org/pdf/2104.03577.pdf
        # optim = SGD(learning_rate=0.002, momentum=0.99)

        self.model.compile(optimizer=optim,
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

        self.model.fit(train_generator,
                       validation_data=val_generator,
                       validation_steps=max(math.floor(len(val_generator))/batch_size, 1),
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
