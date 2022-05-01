
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
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from models.interfaces import BaseModel
from custom_train_objects import get_optimizer, get_loss, get_metrics
from utils.image import build_gif

def random(distribution,
           shape,
           
           mean     = 0.,
           stddev   = 1.,
           
           minval   = 0.,
           maxval   = 1.,
           
           prct     = 0.2,
           ** kwargs
          ):
    if distribution == 'normal':
        return tf.random.normal(shape, mean = mean, stddev = stddev, ** kwargs)
    elif distribution == 'uniform':
        return tf.random.uniform(shape, minval = minval, maxval = maxval, ** kwargs)
    elif distribution == 'poisson':
        return tf.random.poisson(shape, ** kwargs)
    elif distribution == 'randint':
        return tf.random.uniform(
            shape, minval = minval, maxval = maxval, dtype = tf.int32 ** kwargs
        )
    elif distribution == 'one_hot':
        return tf.one_hot(tf.random.uniform(
            (), minval = minval, maxval = maxval, dtype = tf.int32, ** kwargs
        ), depth = maxval, axis = -1)
    elif distribution == 'multiple_one_hot':
        shape = tuple(shape) + (maxval,)
        return tf.cast(tf.random.uniform(
            shape, minval = 0., maxval = 0., dtype = tf.float32
        ) < prct, tf.int32)

class BaseGAN(BaseModel):
    def __init__(self,
                 noise_size,
                 
                 seed       = 10,
                 distribution   = 'normal',
                 
                 threshold  = 0.5,
                 
                 labels     = None,
                 one_hot_label  = True,
                 **kwargs
                ):
        assert labels is None or isinstance(labels, (list, tuple, dict)), "Unknown labels type : {}\n  {}".format(type(labels), labels)
        
        self.seed   = seed
        self.noise_size = noise_size
        self.distribution   = distribution
        
        self.threshold  = threshold
        
        if labels is None or isinstance(labels, dict):
            self.labels = labels
        else:
            if isinstance(labels[0], (list, tuple)):
                self.labels = {'label_{}'.format(i) : l for i, l in enumerate(labels)}
            else:
                self.labels = {'label' : labels}
        self.one_hot_label = one_hot_label
        
        validation_fakes = self.get_random_input(batch_size = 16, seed = self.seed)
        self.validation_noise, self.validation_labels = validation_fakes
        
        super().__init__(** kwargs)
    
    def init_train_config(self, min_update_accuracy = 0.05, ** kwargs):
        self.min_update_accuracy = min_update_accuracy
        
        super().init_train_config(** kwargs)
    
    @property
    def use_labels(self):
        return self.labels is not None
    
    @property
    def nb_labels(self):
        return [] if not self.use_labels else [len(l) for l in self.labels.values()]
    
    @property
    def distribution_labels(self):
        return 'one_hot' if self.one_hot_label else 'randint'
    
    @property
    def label_input_size(self):
        return [n for n in self.nb_labels] if self.one_hot_label else [1 for _ in self.nb_labels]
    
    @property
    def labels_signature(self):
        return tuple([
            tf.TensorSpec(shape = (None, size), dtype = tf.float32)
            for size in self.label_input_size
        ])
        
    @property
    def call_signature(self):
        noise_signature = tf.TensorSpec(
            shape = (None, self.noise_size), dtype = tf.float32
        )
        if not self.use_labels:
            return noise_signature
        return (noise_signature, ) + self.labels_signature
    
    @property
    def output_signature(self):
        return (self.labels_signature, self.labels_signature)
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            augment_prct = lambda step, ** kwargs: 0.5 / tf.math.log(tf.cast(step + 1, tf.float32)),
            min_update_accuracy   = 0.05
        )
    
    def __str__(self):
        des = super().__str__()
        des += "Noise size : {}\n".format(self.noise_size)
        des += "Label conditionned : {}\n".format(self.use_labels)
        if self.use_labels:
            for k, v in self.labels.items():
                des += "- {} : {}\n".format(k, v)
        return des
    
    def call(self, noise_inputs, training = False):
        generated   = self.generator(noise_inputs, training = training)
        scores      = self.discriminator(generated, training = training)
        return generated, scores
    
    def compile(self, 
                loss            = 'GANLoss',    loss_kwargs = {},
                gen_optimizer   = 'adam',       gen_optimizer_kwargs  = {},
                dis_optimizer   = 'adam',       dis_optimizer_kwargs  = {},
                ** kwargs
               ):
        if hasattr(self, 'gan_loss'):
            logging.warning("Models already compiled !")
            return
        
        loss_kwargs['labels'] = self.labels
        
        loss            = get_loss(loss, ** loss_kwargs)
        gen_optimizer   = get_optimizer(gen_optimizer, ** gen_optimizer_kwargs)
        dis_optimizer   = get_optimizer(dis_optimizer, ** dis_optimizer_kwargs)
        gan_metric      = get_metrics('GANMetric', labels = self.labels)
        
        self.add_loss(loss, name = "gan_loss")
        self.add_optimizer(gen_optimizer, name = 'generator_optimizer')
        self.add_optimizer(dis_optimizer, name = 'discriminator_optimizer')
        self.add_metric(gan_metric, 'gan_metric')
    
    def get_random_input(self, batch_size = None, ** kwargs):
        """ Generate random inputs for `generator` """
        if batch_size is not None:
            if not self.use_labels:
                return tf.stack([
                    self.get_random_input(** kwargs)[0] for _ in range(batch_size)
                ]), ()
            datas = [
                self.get_random_input(** kwargs) for _ in range(batch_size)
            ]
            inputs, outputs = [data[0] for data in datas], [data[1] for data in datas]
            
            inputs = [tf.stack(
                [inp[i] for inp in inputs]
            ) for i in range(len(inputs[0]))]
            
            outputs = [tf.stack(
                [out[i] for out in outputs]
            ) for i in range(len(outputs[0]))]
            
            return inputs, outputs
                
        if not self.use_labels:
            return random(self.distribution, (self.noise_size, ), ** kwargs), ()
        
        noise = random(self.distribution, (self.noise_size, ), ** kwargs)
        
        labels = tuple([
            random(self.distribution_labels, (nb,), minval = 0, maxval = nb, ** kwargs)
            for nb in self.nb_labels
        ])
        
        return (noise, ) + labels, labels
    
    def get_input(self, data):
        raise NotImplementedError()
    
    def decode_scores(self, scores):
        """
            Return scores (and labels) predicted by the `discriminator` model
            
            Arguments :
                - scores    : output of the discriminator model
                    if not use_labels   : reality_score
                    else : [reality_score] + label_scores
                    - reality_score : [batch_size, 1], probability to be a real image
                    - label_scores  : [batch_size, nb_labels_i], softmax probability of each label
                    Note : label_scores is a list of length equal to the number of different labels
            Return :
                if not use_labels :
                    reality : [batch_size] of 1 if the reality_score is higher than the threshold (0 otherwise)
                else :
                    reality + labels where 'labels' are [batch_size] the label which has the highest score
        """
        if self.use_labels:
            assert len(scores) == len(self.labels) + 1
            
            is_real = tf.cast(scores[0] > self.threshold, tf.int32)
            labels  = [tf.argmax(s, axis = -1) for s in scores[1:]]
            return [is_real] + labels

        return tf.cast(scores > self.threshold, tf.int32)
    
    def preprocess_data(self, data):
        fake_inputs, fake_target    = self.get_random_input()
        
        real_inputs, real_target    = self.get_input(data)
        
        return (fake_inputs, real_inputs), (fake_target, real_target)
    
    def augment_input(self, real_input):
        return real_input
    
    def augment_data(self, inputs, outputs):
        (fake_in, real_in), (fake_out, real_out) = (inputs, outputs)
        
        real_in = self.augment_input(real_in)
        
        return (fake_in, real_in), (fake_out, real_out)
    
    def _get_train_config(self, * args, test_size = 1, ** kwargs):
        return super()._get_train_config(* args, test_size = test_size, ** kwargs)
    
    def train_step(self, batch):
        (fake_inputs, real_inputs), (fake_labels, real_labels) = batch

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            generated, fake_pred    = self(fake_inputs, training = True)
            
            true_pred   = self.discriminator(real_inputs, training = True)
            
            losses  = self.gan_loss(
                [fake_labels, real_labels], [fake_pred, true_pred]
            )
            gen_loss    = losses[1]
            dis_loss    = losses[2]

        gen_variables = self.generator.trainable_variables
        dis_variables = self.discriminator.trainable_variables
        
        gen_grads = gen_tape.gradient(gen_loss, gen_variables)
        self.generator_optimizer.apply_gradients(zip(gen_grads, gen_variables))
        
        if self.use_labels:
            gen_accuracy = tf.reduce_mean(tf.cast(
                fake_pred[0] > self.threshold, tf.float32
            ))
        else:
            gen_accuracy = tf.reduce_mean(tf.cast(fake_pred > self.threshold, tf.float32))
        
        if gen_accuracy > self.min_update_accuracy or tf.random.uniform(()) < 0.5:
            dis_grads = dis_tape.gradient(dis_loss, dis_variables)
            self.discriminator_optimizer.apply_gradients(zip(dis_grads, dis_variables))
        
        self.update_metrics(
            [fake_labels, real_labels], [fake_pred, true_pred]
        )
        return gen_loss + dis_loss
        
    def eval_step(self, batch):
        (fake_inputs, real_inputs), (fake_labels, real_labels) = batch

        generated, fake_pred    = self(fake_inputs, training = False)
        
        true_pred   = self.discriminator(real_inputs, training = False)
            
        return self.update_metrics(
            [fake_labels, real_labels], [fake_pred, true_pred]
        )

    def build_gif(self, ** kwargs):
        gif_filename = os.path.join(self.train_dir, 'training_gif.gif')
        gen_images = os.path.join(self.train_test_dir, '*_gen.png')
        
        kwargs.setdefault('filename', gif_filename)
        return build_gif(self.train_test_dir, img_name = '*_gen.png', ** kwargs)
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config['seed']  = self.seed
        config['noise_size']    = self.noise_size
        config['distribution']  = self.distribution
        
        config['threshold']     = self.threshold
        
        config['labels']        = self.labels
        config['one_hot_label'] = self.one_hot_label
        
        return config
        