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
import cv2
import os
import numpy as np
import tensorflow as tf
from concurrent.futures.thread import ThreadPoolExecutor


class UVAEDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 encoder,
                 image_paths,
                 input_shape,
                 batch_size,
                 latent_dim):
        self.encoder = encoder
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.half_batch_size = batch_size // 2
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0

    def __getitem__(self, index):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        ex = []
        for f in fs:
            img = f.result()
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(img).reshape(self.input_shape)
            ex.append(x)
        ex = np.asarray(ex).reshape((self.batch_size,) + self.input_shape).astype('float32')
        ex = (ex - 127.5) / 127.5
        dx = np.asarray([self.get_z_vector(size=self.latent_dim).reshape((self.latent_dim,)) for _ in range(self.half_batch_size)])
        dx = np.append(dx, np.asarray(self.graph_forward(self.encoder, ex[:self.half_batch_size])).reshape((self.half_batch_size, self.latent_dim)), axis=0).astype('float32')
        dy = np.append(np.ones(shape=(self.half_batch_size, 1)), np.zeros(shape=(self.half_batch_size, 1)), axis=0).astype('float32')
        gan_z_y = np.ones(shape=(self.batch_size, 1), dtype=np.float32)
        return ex, dx, dy, gan_z_y

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    @staticmethod
    def get_z_vector(size):
        z = np.random.normal(loc=-1.0, scale=0.01, size=size)
        z -= np.min(z)
        z /= np.max(z)
        z = z * 2.0 - 1.0
        return z

    def next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def load_image(self, image_path):
        return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR)

