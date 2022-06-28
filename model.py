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
import tensorflow.keras.backend as K


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.encoder = None
        self.decoder = None
        self.discriminator = None
        self.gan = None
        self.latent_dim = latent_dim

    def build(self):
        assert self.input_shape[0] % 32 == 0
        assert self.input_shape[1] % 32 == 0
        encoder_input, z_mean, z_log_var, z = self.build_encoder()
        decoder_input, decoder_output = self.build_decoder()
        discriminator_input, discriminator_output = self.build_discriminator()
        self.encoder = tf.keras.models.Model(encoder_input, [z_mean, z_log_var, z])
        self.decoder = tf.keras.models.Model(decoder_input, decoder_output)
        self.discriminator = tf.keras.models.Model(discriminator_input, discriminator_output)
        vae_output = self.decoder(z)
        self.vae = tf.keras.models.Model(encoder_input, vae_output)
        self.gan = tf.keras.models.Model(encoder_input, self.discriminator(vae_output))

        # self.encoder.save('checkpoints/encoder.h5', include_optimizer=False)
        # self.decoder.save('checkpoints/decoder.h5', include_optimizer=False)
        # self.discriminator.save('checkpoints/discriminator.h5', include_optimizer=False)
        # self.vae.save('checkpoints/vae.h5', include_optimizer=False)
        # self.gan.save('checkpoints/gan.h5', include_optimizer=False)
        # exit(0)
        return self.vae, self.gan, self.encoder, self.decoder, self.discriminator

    # def load(self, model_path):
    #     self.vae = tf.keras.models.load_model(model_path, compile=False)
    #     self.input_shape = self.vae.input_shape[1:]
    #     return self.vae, self.input_shape

    def build_encoder(self):
        encoder_input = tf.keras.layers.Input(shape=self.input_shape)
        x = encoder_input
        x = self.conv2d(x, 16, 3, 2, 'relu')
        x = self.conv2d(x, 32, 3, 2, 'relu')
        x = self.conv2d(x, 64, 3, 2, 'relu')
        x = self.conv2d(x, 128, 3, 2, 'relu')
        x = self.conv2d(x, 256, 3, 2, 'relu')
        x = self.flatten(x)
        z_mean = self.dense(x, self.latent_dim, 'linear')
        z_log_var = self.dense(x, self.latent_dim, 'linear')
        z = self.sampling(z_mean, z_log_var)
        return encoder_input, z_mean, z_log_var, z

    def build_decoder(self):
        target_rows = self.input_shape[0] // 32
        target_cols = self.input_shape[1] // 32
        target_channels = 256
        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = decoder_input
        x = self.dense(x, target_rows * target_cols * target_channels, input_shape=(self.latent_dim,))
        x = self.reshape(x, (target_rows, target_cols, target_channels))
        x = self.conv2d_transpose(x, 256, 3, 2, 'relu')
        x = self.conv2d_transpose(x, 128, 3, 2, 'relu')
        x = self.conv2d_transpose(x, 64,  3, 2, 'relu')
        x = self.conv2d_transpose(x, 32,  3, 2, 'relu')
        x = self.conv2d_transpose(x, 16,  3, 2, 'relu')
        decoder_output = self.conv2d_transpose(x, self.input_shape[-1], 1, 1, 'sigmoid')
        return decoder_input, decoder_output

    def build_discriminator(self):
        discriminator_input = tf.keras.layers.Input(shape=self.input_shape)
        x = discriminator_input
        x = self.conv2d(x, 16, 3, 2, 'leaky')
        x = self.conv2d(x, 32, 3, 2, 'leaky')
        x = self.conv2d(x, 64, 3, 2, 'leaky')
        x = self.conv2d(x, 128, 3, 2, 'leaky')
        x = self.conv2d(x, 256, 3, 2, 'leaky')
        x = self.conv2d(x, 1, 1, 1, 'sigmoid')
        discriminator_output = self.gap(x)
        return discriminator_input, discriminator_output

    def conv2d(self, x, filters, kernel_size, strides=1, activation='relu', input_shape=()):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            use_bias=False,
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            input_shape=input_shape)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        else:
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

    def conv2d_transpose(self, x, filters, kernel_size, strides=1, activation='relu', input_shape=()):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            use_bias=False,
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            input_shape=input_shape)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        else:
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

    def dense(self, x, units, activation='relu', input_shape=()):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=False,
            kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        else:
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

    def sampling(self, z_mean, z_log_var):
        return tf.keras.layers.Lambda(function=self.sampling_function)([z_mean, z_log_var])

    def sampling_function(self, args):
        z_mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=1.0)
        return z_mean + K.exp(log_var * 0.5) * epsilon

    def reshape(self, x, target_shape):
        return tf.keras.layers.Reshape(target_shape=target_shape)(x)

    def flatten(self, x):
        return tf.keras.layers.Flatten()(x)

    def gap(self, x):
        return tf.keras.layers.GlobalAveragePooling2D()(x)

    # def save(self, path, iteration_count, loss):
    #     self.vae.save(f'{path}/ae_{iteration_count}_iter_{loss:.4f}_loss.h5', include_optimizer=False)

    def summary(self):
        self.encoder.summary()
        print()
        self.decoder.summary()
        print()
        self.discriminator.summary()

