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
import matplotlib.pyplot as plt

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
                 lr=0.0005,
                 batch_size=32,
                 latent_dim=128,
                 iterations=100000,
                 validation_split=0.2,
                 validation_image_path='',
                 checkpoint_path='checkpoints',
                 pretrained_model_path='',
                 view_grid_size=4,
                 training_view=False):
        self.lr = lr
        self.iterations = iterations
        self.training_view = training_view
        self.live_view_previous_time = time()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.checkpoint_path = checkpoint_path
        self.view_grid_size = view_grid_size
        plt.style.use(['dark_background'])
        plt.tight_layout(0.5)
        self.fig, _ = plt.subplots()

        self.model = Model(input_shape=input_shape, latent_dim=self.latent_dim)
        self.encoder, self.decoder, self.z_discriminator, self.d_discriminator, self.vae, self.z_gan, self.d_gan = self.model.build()
        # if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
        #     print(f'\npretrained model path : {[pretrained_model_path]}')
        #     self.decoder = tf.keras.models.load_model(pretrained_model_path, compile=False)
        #     print(f'input_shape : {self.input_shape}')

        if validation_image_path != '':
            self.train_image_paths, _ = self.init_image_paths(train_image_path)
            self.validation_image_paths, _ = self.init_image_paths(validation_image_path)
        elif validation_split > 0.0:
            self.train_image_paths, self.validation_image_paths = self.init_image_paths(train_image_path, validation_split=validation_split)
        self.train_data_generator = UVAEDataGenerator(
            encoder=self.encoder,
            decoder=self.decoder,
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            batch_size=batch_size,
            latent_dim=self.latent_dim)
        self.validation_data_generator = UVAEDataGenerator(
            encoder=self.encoder,
            decoder=self.decoder,
            image_paths=self.validation_image_paths,
            input_shape=input_shape,
            batch_size=batch_size,
            latent_dim=self.latent_dim)
        self.validation_data_generator_one_batch = UVAEDataGenerator(
            encoder=self.encoder,
            decoder=self.decoder,
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

    def train_step_e(self, model, optimizer, x, variance):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_mean = K.square(tf.reduce_mean(y_pred))
            loss_var = K.square(variance - tf.math.reduce_variance(y_pred))
            loss = (loss_mean + loss_var) * 0.5
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def train_step(self, model, optimizer, batch_x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(batch_x, training=True)
            loss = K.square(y_true - y_pred)
            loss = tf.reduce_mean(loss, axis=0)
            mean_loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mean_loss

    def evaluate(self, generator):
        loss_sum = 0.0
        for ex, ae_y, ae_mask in tqdm(generator):
            y = self.graph_forward(self.vae, ex)
            loss_sum += np.mean(np.abs(ae_y[0] - y[0]))
        return float(loss_sum / len(generator))

    def calculate_variance(self, latent_dim):
        var_sum = 0.0
        for _ in tqdm(range(10000)):
            z = UVAEDataGenerator.get_z_vector(size=latent_dim)
            var_sum += np.var(z)
        var = var_sum / 10000.0
        print(f'variance : {var:.4f}')
        return var

    def train(self):
        iteration_count = 0
        optimizer_e =     tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        optimizer_z_d =   tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        optimizer_d_d =   tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        optimizer_vae =   tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        optimizer_z_gan = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        optimizer_d_gan = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)

        train_step_e = tf.function(self.train_step_e)
        train_step_z_d = tf.function(self.train_step)
        train_step_d_d = tf.function(self.train_step)
        train_step_vae = tf.function(self.train_step)
        train_step_z_gan = tf.function(self.train_step)
        train_step_d_gan = tf.function(self.train_step)
        variance = tf.constant(self.calculate_variance(latent_dim=self.latent_dim), dtype=tf.dtypes.float32)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        while True:
            for ex, z_dx, z_dy, d_dx, d_dy, z_gan_y, d_gan_x, d_gan_y in self.train_data_generator:
                iteration_count += 1
                distribution_loss = 0.0
                distribution_loss = self.train_step_e(self.encoder, optimizer_e, ex, variance)
                reconstruction_loss = train_step_vae(self.vae, optimizer_vae, ex, ex)

                z_discriminator_loss = 0.0
                z_adversarial_loss = 0.0
                self.z_discriminator.trainable = True
                z_discriminator_loss = train_step_z_d(self.z_discriminator, optimizer_z_d, z_dx, z_dy)
                self.z_discriminator.trainable = False
                z_adversarial_loss = train_step_z_gan(self.z_gan, optimizer_z_gan, ex, z_gan_y)

                d_adversarial_loss = 0.0
                d_discriminator_loss = 0.0
                self.d_discriminator.trainable = True
                d_discriminator_loss = train_step_d_d(self.d_discriminator, optimizer_d_d, d_dx, d_dy)
                self.d_discriminator.trainable = False
                d_adversarial_loss = train_step_d_gan(self.d_gan, optimizer_d_gan, d_gan_x, d_gan_y)

                print(f'\r[iteration count : {iteration_count:6d}] reconstruction_loss => {reconstruction_loss:.4f}, distribution_loss => {distribution_loss:.4f}, z_discriminator_loss => {z_discriminator_loss:.4f}, z_adversarial_loss => {z_adversarial_loss:.4f}, d_discriminator_loss => {d_discriminator_loss:.4f}, d_adversarial_loss => {d_adversarial_loss:.4f}', end='\t')
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

    def generate(self):
        while True:
            generated_image = self.generate_random_image()
            cv2.imshow('generated image', generated_image)
            key = cv2.waitKey(0)
            if key == 27:
                break

    def generate_random_image(self, size=1):
        z = np.asarray([UVAEDataGenerator.get_z_vector(size=self.latent_dim) for _ in range(size)]).astype('float32')
        y = self.graph_forward(self.decoder, z)
        y = (y * 127.5) + 127.5
        generated_images = np.clip(np.asarray(y).reshape((size,) + self.input_shape), 0.0, 255.0).astype('uint8')
        return generated_images[0] if size == 1 else generated_images

    def generate_latent_space_2d(self, split_size=10):
        assert split_size > 1
        assert self.latent_dim == 2
        space = np.linspace(-1.0, 1.0, split_size)
        z = []
        for i in range(split_size):
            for j in range(split_size):
                z.append([space[i], space[j]])
        z = np.asarray(z).reshape((split_size * split_size, 2)).astype('float32')
        y = self.graph_forward(self.decoder, z)
        y = (y * 127.5) + 127.5
        generated_images = np.clip(np.asarray(y).reshape((split_size * split_size,) + self.input_shape), 0.0, 255.0).astype('uint8')
        return generated_images

    def predict(self, img):
        img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
        x = np.asarray(img).reshape((1,) + self.input_shape).astype('float32')
        x = (x - 127.5) / 127.5
        z = np.asarray(self.graph_forward(self.encoder, x)).reshape(-1)
        # with np.printoptions(precision=2, suppress=True):
        #     print(f'z : {z[0]}')
        y = self.graph_forward(self.vae, x)
        y = np.asarray(y).reshape(self.input_shape)
        y = (y * 127.5) + 127.5
        decoded_img = np.clip(y, 0.0, 255.0).astype('uint8')
        return img, decoded_img, z

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
                img, output_image, _ = self.predict(img)
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

    def make_border(self, img, size=5):
        return cv2.copyMakeBorder(img, size, size, size, size, None, value=192) 

    def make_z_distribution_image(self, z):
        plt.clf()
        plt.hist(z, bins=self.latent_dim if self.latent_dim < 20 else 20)
        # plt.plot(z)
        self.fig.canvas.draw()
        width, height = self.fig.canvas.get_width_height()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
        size = self.input_shape[0]
        img = self.resize(img, (size, size))
        if self.input_shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def training_view_function(self):
        """
        During training, the image is forwarded in real time, showing the results are shown.
        """
        cur_time = time()
        if cur_time - self.live_view_previous_time < 3.0:
            return
        self.live_view_previous_time = cur_time
        img_paths = np.random.choice(self.train_image_paths, size=self.view_grid_size, replace=False)
        win_name = 'training view'
        input_shape = self.vae.input_shape[1:]

        decoded_images_cat = None
        for img_path in img_paths:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if input_shape[-1] == 1 else cv2.IMREAD_COLOR)
            img, output_image, z = self.predict(img)
            img = self.resize(img, (input_shape[1], input_shape[0]))
            img, output_image = self.make_border(img), self.make_border(output_image)
            z_image = self.make_border(self.make_z_distribution_image(z))
            img = img.reshape(img.shape + (1,))
            output_image = output_image.reshape(output_image.shape + (1,))
            z_image = z_image.reshape(z_image.shape + (1,))
            imgs = np.concatenate([img, output_image, z_image], axis=1)
            if decoded_images_cat is None:
                decoded_images_cat = imgs
            else:
                decoded_images_cat = np.append(decoded_images_cat, imgs, axis=0)

        generated_images_cat = None
        generated_images = self.generate_random_image(size=self.view_grid_size * self.view_grid_size)
        # generated_images = self.generate_latent_space_2d(split_size=self.view_grid_size)
        for i in range(self.view_grid_size):
            line = None
            for j in range(self.view_grid_size):
                generated_image = self.make_border(generated_images[i * self.view_grid_size + j])
                if line is None:
                    line = generated_image
                else:
                    line = np.append(line, generated_image, axis=1)
            if generated_images_cat is None:
                generated_images_cat = line
            else:
                generated_images_cat = np.append(generated_images_cat, line, axis=0)
        cv2.imshow('decoded_images', decoded_images_cat)
        cv2.imshow('generated_images', generated_images_cat)
        cv2.waitKey(1)
