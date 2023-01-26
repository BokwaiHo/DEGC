import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning
import random
import numpy as np

class Dense(object):
    """ Dense layer with tunable width """
    def __init__(self, name, input_dim, output_dim, action, prune_filter_id, load_checkpoint, dropout=0., act=tf.nn.tanh):
        '''
        Dense layers
        :param name: name of layer
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param action: controlled filter number of this layer
        :param prune_filter_id: the pruned filter ids of this layer
        :param load_checkpoint: the path of trained parameters in last segment
        :param dropout: dropout ratio
        :param act: activation function
        '''
        self.dropout = dropout
        self.action = action
        self.prune_filter_id = prune_filter_id    
        self.load_checkpoint = load_checkpoint
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim

        with tf.variable_scope(name + 'dense'):
            self.dense_weight = tf.get_variable('weights', shape=(input_dim, output_dim),
                                                dtype=tf.float64,
                                                initializer=tf.train.load_variable(self.load_checkpoint, 'model/'+  name + 'dense/' + 'weights'))
        self.prune_and_expand()


    def prune_and_expand(self):
        #prune_and_expand
        self.prune_id_out = random.sample(range(self.output_dim), self.action_out)
        self.reserve_filter_id = filter(lambda x: x not in self.prune_filter_id, range(self.output_dim))
        self.dense_weight = tf.stack([self.dense_weight[:, id] for id in self.reserve_filter_id], axis=1)
        self.dense_weight = tf.Variable(tf.concat([self.dense_weight, tf.truncated_normal((self.input_dim, self.action),stddev=0.01)], axis=1))

        #musk
        self.mask = np.concatenate([np.zeros((self.input_dim, self.output_dim - self.action)), np.ones((self.input_dim, self.action))], axis=1)



    def __call__(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        output = tf.matmul(x, self.dense_weight)

        return self.act(output)