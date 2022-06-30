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
        self.vae = None
        self.z_discriminator = None
        self.d_discriminator = None
        self.z_gan = None
        self.d_gan = None
        self.latent_dim = latent_dim

    def build(self):
        assert self.input_shape[0] % 32 == 0
        assert self.input_shape[1] % 32 == 0
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.decoder_no_noise = tf.keras.models.Sequential(self.decoder.layers[1:])
        self.z_discriminator = self.build_z_discriminator()
        self.d_discriminator = self.build_d_discriminator()
        self.vae = tf.keras.models.Sequential([self.encoder, self.decoder])
        self.z_gan = tf.keras.models.Sequential([self.encoder, self.z_discriminator])
        self.d_gan = tf.keras.models.Sequential([self.decoder, self.d_discriminator])
        return self.encoder, self.decoder_no_noise, self.z_discriminator, self.d_discriminator, self.vae, self.z_gan, self.d_gan

    # def load(self, model_path):
    #     self.vae = tf.keras.models.load_model(model_path, compile=False)
    #     self.input_shape = self.vae.input_shape[1:]
    #     return self.vae, self.input_shape

    def build_encoder(self):
        m = tf.keras.models.Sequential()
        self.conv2d(m, 16,  3, 2, 'relu', input_shape=self.input_shape)
        self.conv2d(m, 32,  3, 2, 'relu')
        self.conv2d(m, 64,  3, 2, 'relu')
        self.conv2d(m, 128, 3, 2, 'relu')
        self.conv2d(m, 256, 3, 2, 'relu')
        self.flatten(m)
        self.dense(m, self.latent_dim, 'tanh')
        return m

    def build_decoder(self):
        m = tf.keras.models.Sequential()
        target_rows = self.input_shape[0] // 32
        target_cols = self.input_shape[1] // 32
        target_channels = 256
        self.noise_layer(m)
        self.dense(m, target_rows * target_cols * target_channels, input_shape=(self.latent_dim,))
        self.reshape(m, (target_rows, target_cols, target_channels))
        self.conv2d_transpose(m, 256, 3, 2, 'relu')
        self.conv2d_transpose(m, 128, 3, 2, 'relu')
        self.conv2d_transpose(m, 64,  3, 2, 'relu')
        self.conv2d_transpose(m, 32,  3, 2, 'relu')
        self.conv2d_transpose(m, 16,  3, 2, 'relu')
        self.conv2d_transpose(m, self.input_shape[-1], 1, 1, 'tanh')
        return m

    def build_z_discriminator(self):
        m = tf.keras.models.Sequential()
        self.dense(m, 256, 'relu', input_shape=(self.latent_dim,))
        self.dense(m, 256, 'relu')
        self.dense(m, 1, 'linear')
        return m

    def build_d_discriminator(self):
        m = tf.keras.models.Sequential()
        self.conv2d(m, 16,  3, 2, 'leaky', input_shape=self.input_shape)
        self.conv2d(m, 32,  3, 2, 'leaky')
        self.conv2d(m, 64,  3, 2, 'leaky')
        self.conv2d(m, 128, 3, 2, 'leaky')
        self.conv2d(m, 256, 3, 2, 'leaky')
        self.conv2d(m, 1, 1, 1, 'linear')
        self.gap(m)
        return m

    def noise_layer(self, m):
        m.add(tf.keras.layers.Lambda(self.noise))

    def noise(self, f):
        import tensorflow.keras.backend as K
        # f *= tf.random.normal(shape=K.shape(f), mean=0.0, stddev=1.0)
        # f *= tf.random.normal(shape=K.shape(f), mean=0.5, stddev=0.5)
        f *= tf.random.uniform(shape=K.shape(f), minval=0.0, maxval=1.0)
        # f += tf.random.uniform(shape=K.shape(f), minval=-1.0, maxval=1.0)
        f = K.clip(f, -1.0, 1.0)
        return f

    def conv2d(self, m, filters, kernel_size, strides=1, activation='relu', alpha=0.2, input_shape=()):
        m.add(tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            input_shape=input_shape))
        self.activation(m, activation)

    def conv2d_transpose(self, m, filters, kernel_size, strides=1, activation='relu', alpha=0.2, input_shape=()):
        m.add(tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            input_shape=input_shape))
        self.activation(m, activation)

    def dense(self, m, units, activation='relu', alpha=0.2, input_shape=()):
        m.add(tf.keras.layers.Dense(
            units=units,
            kernel_initializer='he_normal'))
        self.activation(m, activation)

    def activation(self, m, activation, alpha=0.2):
        if activation == 'leaky':
            m.add(tf.keras.layers.LeakyReLU(alpha=alpha))
        else:
            m.add(tf.keras.layers.Activation(activation=activation))

    def reshape(self, m, target_shape):
        m.add(tf.keras.layers.Reshape(target_shape=target_shape))

    def flatten(self, m):
        m.add(tf.keras.layers.Flatten())

    def gap(self, m):
        m.add(tf.keras.layers.GlobalAveragePooling2D())

    # def save(self, path, iteration_count, loss):
    #     self.vae.save(f'{path}/ae_{iteration_count}_iter_{loss:.4f}_loss.h5', include_optimizer=False)

    def summary(self):
        self.encoder.summary()
        print()
        self.decoder.summary()
        print()
        self.z_discriminator.summary()

