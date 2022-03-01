
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

import tensorflow as tf


class GANLoss(tf.keras.losses.Loss):
    def __init__(self,
                 labels = None,
                 multi_label    = False,
                 scale_gen_loss = False,
                 gen_smoothed_value = 0.,
                 dis_smoothed_value = 0.05,
                 one_sided_smoothing    = True,
                 reduction  = 'none',
                 ** kwargs
                ):
        super().__init__(reduction = 'none', ** kwargs)
        
        self.labels = labels
        self.multi_label    = multi_label
        self.scale_gen_loss = scale_gen_loss
        
        self.gen_smoothed_value = gen_smoothed_value
        self.dis_smoothed_value = dis_smoothed_value
        self.one_sided_smoothing    = one_sided_smoothing
        
        if multi_label:
            self.gen_loss_fn = tf.keras.losses.binary_crossentropy
            self.dis_loss_fn = tf.keras.losses.binary_crossentropy
        else:
            self.gen_loss_fn = tf.keras.losses.categorical_crossentropy
            self.dis_loss_fn = tf.keras.losses.categorical_crossentropy
    
    @property
    def use_labels(self):
        return self.labels is not None
    
    @property
    def metric_names(self):
        basic_losses = ['loss', 'gen_loss', 'dis_loss', 'dis_true_loss', 'dis_fake_loss']
        if not self.use_labels:
            return basic_losses
        return basic_losses + ['gen_reality_loss'] + [
            'gen_{}_loss'.format(label) for label in self.labels
        ] + ['dis_reality_loss'] + [
            'dis_{}_loss'.format(label) for label in self.labels
        ]

    def generator_loss(self, fake_labels, fake_pred):
        if self.labels is None:
            return tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_pred), fake_pred,
                label_smoothing = self.gen_smoothed_value
            )
        
        reality_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(fake_pred[0]), fake_pred[0],
            label_smoothing = self.gen_smoothed_value
        )
        
        labels_loss = [
            self.gen_loss_fn(label_i, pred_i)
            for label_i, pred_i in zip(fake_labels, fake_pred[1:])
        ]
        return [reality_loss] + labels_loss
        
    def true_discriminator_loss(self, true_labels, true_pred):
        if self.labels is None:
            return tf.keras.losses.binary_crossentropy(
                tf.ones_like(true_pred), true_pred,
                label_smoothing = self.dis_smoothed_value
            )
        
        reality_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(true_pred[0]), true_pred[0],
            label_smoothing = self.dis_smoothed_value
        )
        
        labels_loss = [
            self.dis_loss_fn(label_i, pred_i)
            for label_i, pred_i in zip(true_labels, true_pred[1:])
        ]
        return [reality_loss] + labels_loss
    
    def fake_discriminator_loss(self, fake_labels, fake_pred):
        smoothing = 0. if self.one_sided_smoothing else self.dis_smoothed_value
        if self.labels is not None: fake_pred = fake_pred[0]
        
        return tf.keras.losses.binary_crossentropy(
            tf.zeros_like(fake_pred), fake_pred,
            label_smoothing = smoothing
        )
    
    def scale_factor(self, fake_pred, true_pred):
        if self.labels is not None:
            fake_pred, true_pred = fake_pred[0], true_pred[0]
        
        true_mean = tf.reduce_mean(true_pred)
        fake_mean = tf.reduce_mean(fake_pred)

        return tf.abs(true_mean - fake_mean)
    
    def call(self, y_true, y_pred):
        fake_pred, true_pred = y_pred
        fake_labels, true_labels = y_true
        
        gen_loss = self.generator_loss(fake_labels, fake_pred)
        
        dis_true_loss   = self.true_discriminator_loss(true_labels, true_pred)
        dis_fake_loss   = self.fake_discriminator_loss(fake_labels, fake_pred)
        
        if self.scale_gen_loss:
            factor = self.scale_factor(fake_pred, true_pred)
            gen_loss = gen_loss * factor
        
        if not self.use_labels:
            dis_loss = dis_true_loss + dis_fake_loss
            return tf.stack([
                gen_loss + dis_loss, gen_loss, dis_loss, dis_true_loss, dis_fake_loss
            ], 0)
        
        g_loss      = tf.reduce_sum(tf.stack(gen_loss, -1), axis = -1)
        d_true_loss = tf.reduce_sum(tf.stack(dis_true_loss, -1), axis = -1)
        
        d_loss      = d_true_loss + dis_fake_loss
        loss = d_loss + g_loss
        
        return tf.stack(
            [loss, g_loss, d_loss, d_true_loss, dis_fake_loss] + gen_loss + dis_true_loss, 0
        )
        
    def get_config(self):
        config = super().get_config()
        config['labels']    = self.labels
        config['multi_label']       = self.multi_label
        config['gen_smoothed_value']    = self.gen_smoothed_value
        config['dis_smoothed_value']    = self.dis_smoothed_value
        config['one_sided_smoothing']   = self.one_sided_smoothing
        return config
