
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


class GANMetric(tf.keras.metrics.Metric):
    def __init__(self,
                 labels = None,
                 multi_label    = False,
                 threshold  = 0.5,
                 ** kwargs
                ):
        super().__init__(** kwargs)
        
        self.labels = labels
        self.multi_label    = multi_label
        self.threshold      = threshold
        
        self.batches        = self.add_weight('batches', initializer = 'zeros')
        self.gen_accuracy   = self.add_weight('gen_accuracy', initializer = 'zeros')
        self.dis_accuracy   = self.add_weight('dis_accuracy', initializer = 'zeros')
        
        if labels is not None:
            for label in labels:
                gen_name, dis_name = 'gen_{}'.format(label), 'dis_{}'.format(label)
                setattr(self, gen_name, self.add_weight(gen_name, initializer = 'zeros'))
                setattr(self, dis_name, self.add_weight(dis_name, initializer = 'zeros'))
        
        if multi_label:
            self.gen_metric_fn = tf.keras.metrics.binary_accuracy
            self.dis_metric_fn = tf.keras.metrics.binary_accuracy
        else:
            self.gen_metric_fn = tf.keras.metrics.categorical_accuracy
            self.dis_metric_fn = tf.keras.metrics.categorical_accuracy
    
    @property
    def use_labels(self):
        return self.labels is not None
    
    @property
    def metric_names(self):
        base_metrics = ['gen_realism', 'dis_true_accuracy']
        
        if not self.use_labels:
            return base_metrics
        return base_metrics + [
            'gen_{}'.format(label) for label in self.labels
        ] + [
            'dis_{}'.format(label) for label in self.labels
        ]

    def generator_metric(self, fake_labels, fake_pred):
        if self.labels is None:
            return tf.reduce_sum(tf.cast(fake_pred > self.threshold, tf.float32))
        
        realism = tf.reduce_sum(tf.cast(fake_pred[0] > self.threshold, tf.float32))
        
        labels = [
            tf.reduce_sum(self.gen_metric_fn(label_i, pred_i))
            for label_i, pred_i in zip(fake_labels, fake_pred[1:])
        ]
        return [realism] + labels
        
    def discriminator_metric(self, true_labels, true_pred):
        if self.labels is None:
            return tf.reduce_sum(tf.cast(true_pred > self.threshold, tf.float32))
        
        realism = tf.reduce_sum(tf.cast(true_pred[0] > self.threshold, tf.float32))
        
        labels = [
            tf.reduce_sum(self.dis_metric_fn(label_i, pred_i))
            for label_i, pred_i in zip(true_labels, true_pred[1:])
        ]
        return [realism] + labels
    
    def update_state(self, y_true, y_pred):
        fake_pred, true_pred = y_pred
        fake_labels, true_labels = y_true
        
        gen_metric = self.generator_metric(fake_labels, fake_pred)
        dis_metric = self.discriminator_metric(true_labels, true_pred)
        
        if not self.use_labels:
            self.batches.assign_add(tf.cast(tf.shape(fake_pred)[0], tf.float32))
            self.gen_accuracy.assign_add(gen_metric)
            self.dis_accuracy.assign_add(dis_metric)
            return
        
        
        self.batches.assign_add(tf.cast(tf.shape(fake_pred[0])[0], tf.float32))
        self.gen_accuracy.assign_add(gen_metric[0])
        self.dis_accuracy.assign_add(dis_metric[0])
        
        for i, label in enumerate(self.labels):
            gen_name, dis_name = 'gen_{}'.format(label), 'dis_{}'.format(label)
            getattr(self, gen_name).assign_add(gen_metric[i+1])
            getattr(self, dis_name).assign_add(dis_metric[i+1])
    
    def result(self):
        if not self.use_labels:
            return self.gen_accuracy / self.batches, self.dis_accuracy / self.batches
        
        gen_accuracy = self.gen_accuracy / self.batches
        dis_accuracy = self.dis_accuracy / self.batches
        
        gen_metrics, dis_metrics = [], []
        for i, label in enumerate(self.labels):
            gen_name, dis_name = 'gen_{}'.format(label), 'dis_{}'.format(label)
            gen_metrics.append(getattr(self, gen_name) / self.batches)
            gen_metrics.append(getattr(self, dis_name) / self.batches)
        
        return [gen_accuracy, dis_accuracy] + gen_metrics + dis_metrics
        
    def get_config(self):
        config = super().get_config()
        config['labels']    = self.labels
        config['multi_label']   = self.multi_label
        config['threshold']     = self.threshold
        return config
