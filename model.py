"""
Authors : inzapp

Github url : https://github.com/inzapp/uvae

Copyright 2022 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import numpy as np
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.encoder = None
        self.decoder = None
        self.ae = None
        self.discriminator = None
        self.gan = None
        self.latent_dim = latent_dim

    def build(self):
        assert self.input_shape[0] % 32 == 0
        assert self.input_shape[1] % 32 == 0
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.discriminator = self.build_discriminator()
        self.ae = tf.keras.models.Sequential([self.encoder, self.decoder])
        self.gan = tf.keras.models.Sequential([self.encoder, self.discriminator])
        return self.ae, self.gan, self.encoder, self.decoder, self.discriminator

    # def load(self, model_path):
    #     self.ae = tf.keras.models.load_model(model_path, compile=False)
    #     self.input_shape = self.ae.input_shape[1:]
    #     return self.ae, self.input_shape

    def build_encoder(self):
        m = tf.keras.models.Sequential()
        self.conv2d(m, 16,  3, 2, 'relu', input_shape=self.input_shape)
        self.conv2d(m, 32,  3, 2, 'relu')
        self.conv2d(m, 64,  3, 2, 'relu')
        self.conv2d(m, 128, 3, 2, 'relu')
        self.conv2d(m, 256, 3, 2, 'relu')
        self.gap(m)
        self.dense(m, self.latent_dim, 'tanh')
        return m

    def build_decoder(self):
        m = tf.keras.models.Sequential()
        target_rows = self.input_shape[0] // 32
        target_cols = self.input_shape[1] // 32
        target_channels = 256
        self.dense(m, target_rows * target_cols * target_channels, input_shape=(self.latent_dim,))
        self.reshape(m, (target_rows, target_cols, target_channels))
        self.conv2d_transpose(m, 256, 3, 2, 'relu')
        self.conv2d_transpose(m, 128, 3, 2, 'relu')
        self.conv2d_transpose(m, 64,  3, 2, 'relu')
        self.conv2d_transpose(m, 32,  3, 2, 'relu')
        self.conv2d_transpose(m, 16,  3, 2, 'relu')
        self.conv2d_transpose(m, self.input_shape[-1], 1, 1, 'tanh')
        return m

    def build_discriminator(self):
        m = tf.keras.models.Sequential()
        self.dense(m, 256, 'relu', input_shape=(self.latent_dim,))
        self.dense(m, 256, 'relu')
        self.dense(m, 1, 'linear')
        return m

    def conv2d(self, m, filters, kernel_size, strides=1, activation='relu', input_shape=()):
        m.add(tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            activation=activation,
            input_shape=input_shape))

    def conv2d_transpose(self, m, filters, kernel_size, strides=1, activation='relu', input_shape=()):
        m.add(tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            activation=activation,
            input_shape=input_shape))

    def dense(self, m, units, activation='relu', input_shape=()):
        m.add(tf.keras.layers.Dense(
            units=units,
            kernel_initializer='he_normal',
            activation=activation))

    def reshape(self, m, target_shape):
        m.add(tf.keras.layers.Reshape(target_shape=target_shape))

    def gap(self, m):
        m.add(tf.keras.layers.GlobalAveragePooling2D())

    # def save(self, path, iteration_count, loss):
    #     self.ae.save(f'{path}/ae_{iteration_count}_iter_{loss:.4f}_loss.h5', include_optimizer=False)

    def summary(self):
        self.encoder.summary()
        print()
        self.decoder.summary()
        print()
        self.discriminator.summary()

