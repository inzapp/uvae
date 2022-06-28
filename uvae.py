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
import natsort
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from cv2 import cv2
from glob import glob
from tqdm import tqdm
from time import time
from model import Model
from generator import UVAEDataGenerator


class UniformVectorizedAutoEncoder:
    def __init__(self,
                 train_image_path=None,
                 input_shape=(64, 64, 1),
                 lr=0.001,
                 batch_size=32,
                 latent_dim=16,
                 iterations=100000,
                 validation_split=0.2,
                 validation_image_path='',
                 checkpoint_path='checkpoints',
                 pretrained_model_path='',
                 training_view=False):
        self.lr = lr
        self.iterations = iterations
        self.training_view = training_view
        self.live_view_previous_time = time()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.checkpoint_path = checkpoint_path
        self.view_flag = 1

        self.model = Model(input_shape=input_shape, latent_dim=self.latent_dim)
        # if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
        #     print(f'\npretrained model path : {[pretrained_model_path]}')
        #     self.vae, self.input_shape = self.model.load(pretrained_model_path)
        #     print(f'input_shape : {self.input_shape}')
        # else:
        self.vae, self.gan, self.encoder, self.decoder, self.discriminator = self.model.build()

        if validation_image_path != '':
            self.train_image_paths, _ = self.init_image_paths(train_image_path)
            self.validation_image_paths, _ = self.init_image_paths(validation_image_path)
        elif validation_split > 0.0:
            self.train_image_paths, self.validation_image_paths = self.init_image_paths(train_image_path, validation_split=validation_split)
        self.train_data_generator = UVAEDataGenerator(
            vae=self.vae,
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            batch_size=batch_size,
            latent_dim=self.latent_dim)
        self.validation_data_generator = UVAEDataGenerator(
            vae=self.vae,
            image_paths=self.validation_image_paths,
            input_shape=input_shape,
            batch_size=batch_size,
            latent_dim=self.latent_dim)
        self.validation_data_generator_one_batch = UVAEDataGenerator(
            vae=self.vae,
            image_paths=self.validation_image_paths,
            input_shape=input_shape,
            batch_size=1,
            latent_dim=self.latent_dim)

    def fit(self):
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print(f'validate on {len(self.validation_image_paths)} samples.')
        print('start training')
        self.train()

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    @tf.function
    def train_step_encoder(self, model, optimizer, x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = model(x, training=True)
            loss = -0.5 * (1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)
            mean_loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mean_loss

    @tf.function
    def train_step_vae(self, model, optimizer, x):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            r_loss_factor = 1.0
            # loss = tf.reduce_mean(K.square(x - y_pred), axis=0) * r_loss_factor
            loss = tf.reduce_mean(K.binary_crossentropy(x, y_pred), axis=0)
            mean_loss = tf.reduce_mean(K.square(x - y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mean_loss

    @tf.function
    def train_step_gan(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(K.binary_crossentropy(y_true, y_pred), axis=0)
            mean_loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mean_loss

    def evaluate(self, generator):
        loss_sum = 0.0
        for ae_x, ae_y, ae_mask in tqdm(generator):
            y = self.graph_forward(self.vae, ae_x)
            loss_sum += np.mean(np.abs(ae_y[0] - y[0]))
        return float(loss_sum / len(generator))

    def train(self):
        iteration_count = 0
        optimizer_e = tf.keras.optimizers.Adam(lr=self.lr)
        optimizer_vae = tf.keras.optimizers.Adam(lr=self.lr)
        optimizer_gan = tf.keras.optimizers.Adam(lr=self.lr)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        while True:
            for vae_x, dx, dy, gan_y in self.train_data_generator:
                iteration_count += 1
                # self.discriminator.trainable = True
                kl_loss = self.train_step_encoder(self.encoder, optimizer_e, vae_x)
                reconstruction_loss = self.train_step_vae(self.vae, optimizer_vae, vae_x)
                #self.discriminator.trainable = False
                #adversarial_loss = self.train_step_gan(self.gan, optimizer_gan, vae_x, gan_y)
                adversarial_loss = 0.0
                print(f'\r[iteration count : {iteration_count:6d}] kl_loss => {kl_loss:.4f}, reconstruction_loss => {reconstruction_loss:.4f}, adversarial_loss => {adversarial_loss:.4f}', end='\t')
                if self.training_view:
                    self.training_view_function()
                if iteration_count % 5000 == 0:
                    self.decoder.save(f'{self.checkpoint_path}/decoder_{iteration_count}_iter.h5', include_optimizer=False)
                    pass
                    # loss = self.evaluate(generator=self.validation_data_generator_one_batch)
                    # self.model.save(self.checkpoint_path, iteration_count, loss)
                    # print(f'[{iteration_count} iter] val_loss : {loss:.4f}\n')
                if iteration_count == self.iterations:
                    print('\n\ntrain end successfully')
                    return

    @staticmethod
    def init_image_paths(image_path, validation_split=0.0):
        all_image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        np.random.shuffle(all_image_paths)
        num_train_images = int(len(all_image_paths) * (1.0 - validation_split))
        image_paths = all_image_paths[:num_train_images]
        validation_image_paths = all_image_paths[num_train_images:]
        return image_paths, validation_image_paths

    def resize(self, img, size):
        if img.shape[1] > size[0] or img.shape[0] > size[1]:
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    def generate_random_image(self):
        z = np.random.normal(0.0, 1.0, size=self.latent_dim).reshape((1, self.latent_dim))
        y = self.graph_forward(self.decoder, z)
        # y = (y * 127.5) + 127.5
        y *= 255.0
        generated_image = np.clip(np.asarray(y).reshape(self.input_shape), 0.0, 255.0).astype('uint8')
        return generated_image

    def predict(self, img):
        img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
        x = np.asarray(img).reshape((1,) + self.input_shape).astype('float32')
        # x = (x - 127.5) / 127.5
        x /= 255.0
        z = self.encoder(x, training=False)
        # with np.printoptions(precision=2, suppress=True):
        #     print(f'z : {z[0]}')
        cv2.imshow('random', self.generate_random_image())

        y = self.vae(x, training=False)
        y = np.asarray(y).reshape(self.input_shape).astype('float32')
        # y = (y * 127.5) + 127.5
        y *= 255.0
        decoded_img = np.clip(y, 0.0, 255.0).astype('uint8')
        return img, decoded_img

    def predict_images(self, image_paths):
        """
        Equal to the evaluate function. image paths are required.
        """
        if type(image_paths) is str:
            image_paths = glob(image_paths)
        image_paths = natsort.natsorted(image_paths)
        with tf.device('/cpu:0'):
            for path in image_paths:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR)
                img, output_image = self.predict(img)
                img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
                img = np.asarray(img).reshape(img.shape[:2] + (self.input_shape[-1],))
                cv2.imshow('ae', np.concatenate((img, output_image), axis=1))
                key = cv2.waitKey(0)
                if key == 27:
                    break

    def predict_train_images(self):
        self.predict_images(self.train_image_paths)

    def predict_validation_images(self):
        self.predict_images(self.validation_image_paths)

    def training_view_function(self):
        """
        During training, the image is forwarded in real time, showing the results are shown.
        """
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            if self.view_flag == 1:
                img_path = np.random.choice(self.train_image_paths)
                win_name = 'ae train data'
                self.view_flag = 0
            else:
                img_path = np.random.choice(self.validation_image_paths)
                win_name = 'ae validation data'
                self.view_flag = 1
            input_shape = self.vae.input_shape[1:]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR)
            img, output_image = self.predict(img)
            img = self.resize(img, (input_shape[1], input_shape[0]))
            cv2.imshow(win_name, np.concatenate((img.reshape(input_shape), output_image), axis=1))
            cv2.waitKey(1)
