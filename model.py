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
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.z_discriminator = None
        self.d_discriminator = None
        self.z_gan = None
        self.d_gan = None

    def build(self):
        assert self.input_shape[0] % 32 == 0
        assert self.input_shape[1] % 32 == 0
        encoder_input, z_mean, z_log_var, z = self.build_encoder()
        decoder_input, decoder_output = self.build_decoder()
        self.encoder = tf.keras.models.Model(encoder_input, [z_mean, z_log_var, z])
        self.decoder = tf.keras.models.Model(decoder_input, decoder_output)
        # self.decoder_no_noise = tf.keras.models.Sequential(self.decoder.layers[1:])
        z_discriminator_input, z_discriminator_output = self.build_z_discriminator()
        d_discriminator_input, d_discriminator_output = self.build_d_discriminator()
        self.z_discriminator = tf.keras.models.Model(z_discriminator_input, z_discriminator_output)
        self.d_discriminator = tf.keras.models.Model(d_discriminator_input, d_discriminator_output)
        vae_output = self.decoder(z)
        self.vae = tf.keras.models.Model(encoder_input, vae_output)
        self.z_gan = tf.keras.models.Model(encoder_input, self.z_discriminator(z))
        self.d_gan = tf.keras.models.Model(encoder_input, self.d_discriminator(vae_output))
        # return self.encoder, self.decoder_no_noise, self.z_discriminator, self.d_discriminator, self.vae, self.z_gan, self.d_gan

        # self.encoder.save('checkpoints/encoder.h5', include_optimizer=False)
        # self.decoder.save('checkpoints/decoder.h5', include_optimizer=False)
        # self.z_discriminator.save('checkpoints/z_discriminator.h5', include_optimizer=False)
        # self.d_discriminator.save('checkpoints/d_discriminator.h5', include_optimizer=False)
        # self.vae.save('checkpoints/vae.h5', include_optimizer=False)
        # self.z_gan.save('checkpoints/z_gan.h5', include_optimizer=False)
        # self.d_gan.save('checkpoints/d_gan.h5', include_optimizer=False)
        return self.encoder, self.decoder, self.z_discriminator, self.d_discriminator, self.vae, self.z_gan, self.d_gan

    # def load(self, model_path):
    #     self.vae = tf.keras.models.load_model(model_path, compile=False)
    #     self.input_shape = self.vae.input_shape[1:]
    #     return self.vae, self.input_shape

    def build_encoder(self):
        encoder_input = tf.keras.layers.Input(shape=self.input_shape)
        x = encoder_input
        x = self.conv2d(x, 16,  3, 2, 'relu')
        x = self.conv2d(x, 32,  3, 2, 'relu')
        x = self.conv2d(x, 64,  3, 2, 'relu')
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
        # x = self.noise_layer(x)
        x = self.dense(x, target_rows * target_cols * target_channels)
        x = self.reshape(x, (target_rows, target_cols, target_channels))
        x = self.conv2d_transpose(x, 256, 3, 2, 'relu')
        x = self.conv2d_transpose(x, 128, 3, 2, 'relu')
        x = self.conv2d_transpose(x, 64,  3, 2, 'relu')
        x = self.conv2d_transpose(x, 32,  3, 2, 'relu')
        x = self.conv2d_transpose(x, 16,  3, 2, 'relu')
        decoder_output = self.conv2d_transpose(x, self.input_shape[-1], 1, 1, 'tanh')
        return decoder_input, decoder_output

    def build_z_discriminator(self):
        z_discriminator_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = z_discriminator_input
        x = self.dense(x, 256, 'tanh')
        x = self.dense(x, 256, 'tanh')
        z_discriminator_output = self.dense(x, 1, 'linear')
        return z_discriminator_input, z_discriminator_output

    def build_d_discriminator(self):
        d_discriminator_input = tf.keras.layers.Input(shape=self.input_shape)
        x = d_discriminator_input
        x = self.conv2d(x, 16,  3, 2, 'leaky')
        x = self.conv2d(x, 32,  3, 2, 'leaky')
        x = self.conv2d(x, 64,  3, 2, 'leaky')
        x = self.conv2d(x, 128, 3, 2, 'leaky')
        x = self.conv2d(x, 256, 3, 2, 'leaky')
        x = self.conv2d(x, 1, 1, 1, 'linear')
        d_discriminator_output = self.gap(x)
        return d_discriminator_input, d_discriminator_output

    def sampling(self, z_mean, z_log_var):
        def function(args):
            z_mean, log_var = args
            epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=1.0)
            return z_mean + K.exp(log_var * 0.5) * epsilon
        return tf.keras.layers.Lambda(function=function)([z_mean, z_log_var])

    def noise_layer(self, x):
        def function(f):
            # f *= tf.random.normal(shape=K.shape(f), mean=0.0, stddev=1.0)
            # f *= tf.random.normal(shape=K.shape(f), mean=0.5, stddev=0.5)
            f *= tf.random.uniform(shape=K.shape(f), minval=0.0, maxval=1.0)
            f += tf.random.uniform(shape=K.shape(f), minval=-0.1, maxval=0.1)
            f = K.clip(f, -1.0, 1.0)
            return f
        return tf.keras.layers.Lambda(function=function)(x)

    def conv2d(self, x, filters, kernel_size, strides=1, activation='relu', alpha=0.2):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer='he_normal')(x)
        return self.activation(x, activation)

    def conv2d_transpose(self, x, filters, kernel_size, strides=1, activation='relu', alpha=0.2):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            kernel_initializer='he_normal')(x)
        return self.activation(x, activation)

    def dense(self, x, units, activation='relu', alpha=0.2):
        x = tf.keras.layers.Dense(
            units=units,
            kernel_initializer='he_normal')(x)
        return self.activation(x, activation)

    def activation(self, x, activation, alpha=0.2):
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
        else:
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

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
        self.z_discriminator.summary()

