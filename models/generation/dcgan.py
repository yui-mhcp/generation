
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf

from math import sqrt

from utils import plot_multiple
from utils.image import load_image
from models.generation.base_gan import BaseGAN

class DCGAN(BaseGAN):
    def __init__(self,
                 image_shape,
                 * args,
                 channels   = None,
                 tanh_normalize = True,
                 ** kwargs
                ):
        if isinstance(image_shape, int):
            assert channels is not None, "You must specify channels or image_shape as (h,w,c) !"
            image_shape = (image_shape, image_shape, channels)
        
        self.image_shape = tuple(image_shape)
        self.tanh_normalize = tanh_normalize
        
        super().__init__(* args, ** kwargs)
        
                        
    def _build_model(self, **kwargs):
        dis_activation = 'sigmoid'
        if self.use_labels:
            dis_activation = ['sigmoid'] + ['softmax' for _ in self.nb_labels]
        
        noise_size = self.noise_size
        if self.use_labels: noise_size = [noise_size] + self.nb_labels

        super()._build_model(
            generator       = {
                'architecture_name' : 'simple_generator',
                'noise_size'    : noise_size,
                'output_shape'  : self.image_shape,
                'final_activation'  : self.final_activation,
                ** kwargs.get('gen_kwargs', {})
            },
            discriminator   = {
                'architecture_name' : 'simple_cnn',
                'input_shape'   : self.image_shape,
                'final_activation'  : dis_activation,
                'output_shape'  : 1 if not self.use_labels else [1] + self.nb_labels,
                ** kwargs.get('dis_kwargs', {})
            }
        )
        
    
    @property
    def input_signature(self):
        return (
            self.call_signature,
            tf.TensorSpec(shape = (None,) + self.image_shape, dtype = tf.float32)
        )
    
    @property
    def final_activation(self):
        return 'tanh' if self.tanh_normalize else 'sigmoid'
    
    @property
    def img_value_range(self):
        return [-1, 1] if self.tanh_normalize else [0, 1]                
        
    def __str__(self):
        des = super().__str__()
        des += "Image shape : {}\n".format(self.image_shape)
        return des
    
    def normalize_image(self, image):
        """
            Arguments :
                - image : filename / raw image
            Return :
                - normalized image  : image with value in range `self.img_value_range`
        """
        image = load_image(image, target_shape = self.image_shape)
        
        return image * 2 - 1. if self.tanh_normalize else image
        
    def de_normalize_image(self, image):
        """
            Arguments :
                - image : generated image in range `self.img_value_range`
            Return :
                - normalized image  : image in range [0, 1]
        """
        return image / 2. + 0.5 if self.tanh_normalize else image
    
    def get_input(self, data):
        image = self.normalize_image(data)
        
        if not self.use_labels:
            return image, ()
        
        return image, (tf.one_hot(data['label'], 10), )
    
    def augment_input(self, real_input):
        return tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: real_input + tf.random.normal(tf.shape(real_input)) / 10.,
            lambda: real_input
        )
    
    def predict_with_target(self, batch, noise = None, fake_labels = None,
                            epoch = None, step = None, prefix = None, directory = None):
        fake_inputs = self.validation_noise
        
        title = "Generated images"
        if epoch is not None: title += " at epoch {}".format(epoch)
        
        filename = prefix + '_gen.png'
        
        if directory is not None:
            filename = os.path.join(directory, filename)
        
        self.predict(noise = fake_inputs, title = title, filename = filename)
    
    def predict(self,
                n       = None,
                noise   = None,
                only_accepted   = False, 
                with_scores     = True,
                with_predicted_label    = False,
                ** kwargs
               ):
        if noise is None:
            noise, _ = self.get_random_input(batch_size = n)

        images, scores = self(noise, training = False)
        realism_scores = tf.reshape(scores, [-1]) if not self.use_labels else tf.reshape(scores[0], [-1])
        
        outputs = self.decode_scores(scores)
        
        if only_accepted:
            is_valid = labels if not self.use_labels else labels[0]
            
            images = [img for img, v in zip(images, is_valid) if v]
            scores = [s for s, v in zip(scores, is_valid) if v]
        
        images = self.de_normalize_image(images)
        
        labels = []
        if self.labels is not None:
            for i in range(len(realism_scores)):
                labels.append({
                    label : (outputs[j+1][i], np.argmax(noise[j+1][i]))
                    for j, label in enumerate(self.labels.keys())
                })
        else:
            labels = [{} for _ in range(len(realism_scores))]
        
        names = [
            'Real {:.2f}{}'.format(s * 100, ''.join([
                '\n{} : {} (target {})'.format(label, label_p, label_t)
                for label, (label_p, label_t) in labels_i.items()
            ]))
            for i, (s, labels_i) in enumerate(zip(realism_scores, labels))
        ]

        infos = [
            (name, self.de_normalize_image(img)) for img, name in zip(images, names)
        ]
        
        if len(infos) > 0:
            kwargs.setdefault('ncols', int(sqrt(len(infos))))
            plot_multiple(
                * infos, plot_type = 'imshow', ** kwargs
            )
        
        return images, scores, outputs
        
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config['image_shape']    = self.image_shape
        config['tanh_normalize']    = self.tanh_normalize
        
        return config
        