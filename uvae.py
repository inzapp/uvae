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
        self.z_losses = []
        plt.style.use(['dark_background'])
        plt.tight_layout(0.5)
        self.fig, _ = plt.subplots()
        if self.latent_dim == -1:
            self.latent_dim = self.input_shape[0] // 32 * self.input_shape[1] // 32 * 256

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

    @tf.function
    def train_step_distribution(self, model, optimizer, x, mean, var, std):
        with tf.GradientTape() as tape:
            z = model(x, training=True)
            loss_mean = K.square(mean - tf.reduce_mean(z))
            loss_var = K.square(var - tf.math.reduce_variance(z))
            loss_std = K.square(std - tf.math.reduce_std(z))
            loss = (loss_mean + loss_var + loss_std) / 3.0
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def train_step_distance(self, model, optimizer, x):
        with tf.GradientTape() as tape:
            half_batch_size = K.cast(K.shape(x)[0] / 2, dtype=tf.int32)
            z = model(x, training=True)
            loss = -tf.reduce_mean(K.square(z[half_batch_size:] - z[:half_batch_size]))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def train_step_mse(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(K.square(y_true - model(x, training=True)))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def train_step_vae(self, model, optimizer, x, y_true):
        def softclip(tensor, min_val):
            return min_val + K.softplus(tensor - min_val)
        def gaussian_nll(mu, log_sigma, x):
            return 0.5 * K.square((x - mu) / K.exp(log_sigma)) + log_sigma + 0.5 * K.log(np.pi * 2.0)
        with tf.GradientTape() as tape:
            batch_size = K.cast(K.shape(x)[0], dtype=tf.float32)
            y_pred, mu, log_var = model(x, training=True)
            log_sigma = K.log(K.sqrt(tf.reduce_mean(K.square(y_true - y_pred))))
            log_sigma = softclip(log_sigma, -6.0)
            reconstruction_loss = tf.reduce_sum(gaussian_nll(y_pred, log_sigma, y_true)) / batch_size
            kl_loss = -0.5 * tf.reduce_sum((1.0 + log_var - K.square(mu) - K.exp(log_var))) / batch_size
            loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return reconstruction_loss, kl_loss

    def calculate_mean_var_std(self, latent_dim, iteration=10000):
        mean_sum, var_sum, std_sum = 0.0, 0.0, 0.0
        for _ in tqdm(range(iteration)):
            z = UVAEDataGenerator.get_z_vector(size=latent_dim)
            mean_sum += np.mean(z)
            var_sum += np.var(z)
            std_sum += np.std(z)
        mean = mean_sum / float(iteration)
        var = var_sum / float(iteration)
        std = std_sum / float(iteration)
        print(f'mean, variance, std: {mean:.4f}, {var:.4f}, {std:.4f}\n')
        mean = tf.constant(mean, dtype=tf.dtypes.float32)
        var = tf.constant(var, dtype=tf.dtypes.float32)
        std = tf.constant(std, dtype=tf.dtypes.float32)
        return mean, var, std

    def get_z_loss_diff(self, z_d_loss, z_a_loss):
        self.z_losses.append(abs(z_d_loss - z_a_loss))
        if len(self.z_losses) > 100:
            self.z_losses.pop(0)
        return np.mean(self.z_losses)

    def train(self):
        iteration_count = 0
        optimizer_e =     tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        optimizer_z_d =   tf.keras.optimizers.Adam(lr=self.lr * 0.2, beta_1=0.5)
        optimizer_d_d =   tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        optimizer_vae =   tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)
        optimizer_z_gan = tf.keras.optimizers.Adam(lr=self.lr * 0.2, beta_1=0.5)
        optimizer_d_gan = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.5)

        # optimizer_e =     tf.keras.optimizers.RMSprop(lr=self.lr)
        # optimizer_z_d =   tf.keras.optimizers.RMSprop(lr=self.lr)
        # optimizer_d_d =   tf.keras.optimizers.RMSprop(lr=self.lr)
        # optimizer_vae =   tf.keras.optimizers.RMSprop(lr=self.lr)
        # optimizer_z_gan = tf.keras.optimizers.RMSprop(lr=self.lr)
        # optimizer_d_gan = tf.keras.optimizers.RMSprop(lr=self.lr)

        train_step_z_d = tf.function(self.train_step_mse)
        train_step_d_d = tf.function(self.train_step_mse)
        train_step_ae = tf.function(self.train_step_mse)
        # train_step_ae = tf.function(self.train_step_ae)
        train_step_z_gan = tf.function(self.train_step_mse)
        train_step_d_gan = tf.function(self.train_step_mse)
        mean, var, std = self.calculate_mean_var_std(latent_dim=self.latent_dim)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        train_z_d = True
        while True:
            # for ex, z_dx, z_dy, d_dx, d_dy, z_gan_y, d_gan_x, d_gan_y in self.train_data_generator:
            for ex, z_dx, z_dy, fake_label in self.train_data_generator:
                iteration_count += 1

                distribution_loss = 0.0
                reconstruction_loss = 0.0
                distance_loss = 0.0
                kl_loss = 0.0
                # distribution_loss = self.train_step_distribution(self.encoder, optimizer_e, ex, mean, var, std)
                # distance_loss = self.train_step_distance(self.encoder, optimizer_e, ex)
                # reconstruction_loss = train_step_ae(self.vae, optimizer_vae, ex, ex)
                reconstruction_loss, kl_loss = self.train_step_vae(self.vae, optimizer_vae, ex, ex)

                # z_discriminator_loss = 0.0
                # z_adversarial_loss = 0.0
                # z_loss_diff = 0.0
                # if train_z_d:
                #     self.z_discriminator.trainable = True
                #     z_discriminator_loss = train_step_z_d(self.z_discriminator, optimizer_z_d, z_dx, z_dy)
                # self.z_discriminator.trainable = False
                # z_adversarial_loss = train_step_z_gan(self.z_gan, optimizer_z_gan, ex, fake_label)
                # z_loss_diff = self.get_z_loss_diff(z_discriminator_loss, z_adversarial_loss)
                # if z_loss_diff < 0.05:
                #     train_z_d = False

                d_adversarial_loss = 0.0
                d_discriminator_loss = 0.0
                # self.d_discriminator.trainable = True
                # d_discriminator_loss = train_step_d_d(self.d_discriminator, optimizer_d_d, d_dx, d_dy)
                # self.d_discriminator.trainable = False
                # d_adversarial_loss = train_step_d_gan(self.d_gan, optimizer_d_gan, d_gan_x, d_gan_y)

                if kl_loss > 0.0:
                    print(f'[iteration count : {iteration_count:6d}] reconstruction_loss : {reconstruction_loss:.4f}, kl_loss : {kl_loss:.4f}')
                else:
                    print(f'[iteration count : {iteration_count:6d}] reconstruction_loss : {reconstruction_loss:.4f}, distribution_loss : {distribution_loss:.4f}, distance_loss : {distance_loss:.4f}, z_discriminator_loss : {z_discriminator_loss:.4f}, z_adversarial_loss : {z_adversarial_loss:.4f}, z_loss_diff : {z_loss_diff:.4f}')
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
        # z = np.zeros(shape=((size, self.latent_dim)), dtype=np.float32)
        y = np.asarray(self.graph_forward(self.decoder, z))[0]
        y = UVAEDataGenerator.denormalize(y)
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
        y = np.asarray(self.graph_forward(self.decoder, z))
        y = UVAEDataGenerator.denormalize(y)
        generated_images = np.clip(np.asarray(y).reshape((split_size * split_size,) + self.input_shape), 0.0, 255.0).astype('uint8')
        return generated_images

    def predict(self, img):
        img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
        x = np.asarray(img).reshape((1,) + self.input_shape).astype('float32')
        x = UVAEDataGenerator.normalize(x)
        z = np.asarray(self.graph_forward(self.encoder, x))[0].reshape(-1)
        y = np.asarray(self.graph_forward(self.decoder, z.reshape(1, self.latent_dim)))[0].reshape(self.input_shape)
        y = UVAEDataGenerator.denormalize(y)
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

    def show_interpolation(self, frame=100):
        space = np.linspace(-1.0, 1.0, frame)
        for val in space:
            z = np.zeros(shape=(1, self.latent_dim), dtype=np.float32) + val
            y = np.asarray(self.graph_forward(self.decoder, z))[0]
            y = UVAEDataGenerator.denormalize(y)
            generated_image = np.clip(np.asarray(y).reshape(self.input_shape), 0.0, 255.0).astype('uint8')
            cv2.imshow('interpolation', generated_image)
            key = cv2.waitKey(10)
            if key == 27:
                break

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
            # with np.printoptions(precision=2, suppress=True):
            #     print(z)
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
        if self.latent_dim == 2:
            generated_images = self.generate_latent_space_2d(split_size=self.view_grid_size)
        else:
            generated_images = self.generate_random_image(size=self.view_grid_size * self.view_grid_size)
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
        # self.show_interpolation()
        cv2.imshow('decoded_images', decoded_images_cat)
        cv2.imshow('generated_images', generated_images_cat)
        cv2.waitKey(1)
