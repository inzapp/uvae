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
from uvae import UniformVectorizedAutoEncoder

if __name__ == '__main__':
    UniformVectorizedAutoEncoder(
        input_shape=(64, 64, 1),
        train_image_path=r'/train_data/mnist/train',
        validation_image_path=r'/train_data/mnist/validation',
        lr=0.001,
        ae_burn=1000,
        batch_size=32,
        latent_dim=32,
        view_grid_size=4,
        iterations=1000000,
        z_activation='sigmoid',
        z_adversarial_attack=False,
        d_adversarial_attack=False,
        vanilla_vae=False,
        training_view=True).fit()

