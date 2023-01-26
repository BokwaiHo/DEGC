import tensorflow as tf
from models.layers import Dense
import numpy as np

class Dynamic_MyMeanAggregator(object):
    ''' one weight with dimension (input_dim+neigh_dim, output_dim) on the concatenated vectors of
    self and neighbors after mean '''

    def __init__(self, name, num_sample, input_dim, neigh_dim, output_dim, activation, load_checkpoint, input_prune, input_prune_filter_ids, output_prune_filter_ids, dropout=0.2):
        '''
        GCN mean aggregation for neighbor information
        :param name: name of defined aggregator
        :param num_sample: the number of positive neighbors user_user or item_item_graph
        :param input_dim: self_embedding dimension
        :param neigh_dim: neighbor_embedding dimension
        :param output_dim: output embedding dimension
        :param activation: activation function
        :param load_checkpoint: the path of trained parameters in last segment
        :param input_prune: whether pruning and expanding along the input dimension of transformation matrix
        :param input_prune_filter_ids: presampled filter ids of neigh part to be pruned along the input dimension of transformation matrix
        :param output_prune_filter_ids: presampled fiter ids to be prune along the output dimension of transformation matrix
        :param dropout: dropout rate
        '''


        #self.output_weight = tf.get_variable(name + 'output_weights', shape=(input_dim + neigh_dim, output_dim),
        #                                    dtype=tf.float64,
        #                                    initializer=tf.train.load_variable(load_checkpoint, 'model/'+  name + 'output_weights'), trainable=True)
        self.name = name
        #self.output_weight = tf.Variable(initial_value=tf.train.load_variable(load_checkpoint, name + 'output_weight'), 
        #                                 dtype=tf.float64, shape=(input_dim + neigh_dim, output_dim))

        #self.output_weight = tf.Variable(initial_value=tf.train.load_variable(load_checkpoint, name + 'output_weight'), 
        #                                 dtype=tf.float64)
        self.output_weight = tf.train.load_variable(load_checkpoint, name + 'output_weight')
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.act = activation
        self.neigh_dropout = dropout
        self.num_sample = num_sample
        self.neigh_dim = neigh_dim
        self.input_prune = input_prune
        self.input_prune_filter_ids = np.array(input_prune_filter_ids)
        self.output_prune_filter_ids = np.array(output_prune_filter_ids)
        #self.prune_and_expand()
        self.expand()

    def prune_and_expand(self):
        if self.input_prune:
            self.input_prune_filter_ids = self.input_prune_filter_ids + self.input_dim
            self.input_reserve_filter_ids = list(filter(lambda x: x not in self.input_prune_filter_ids, range(self.input_dim + self.neigh_dim)))
            self.output_reserve_filter_ids = list(filter(lambda x: x not in self.output_prune_filter_ids, range(self.output_dim)))

            #prune and expand along the input dimension
            self.output_weight = tf.stack([self.output_weight[i, :] for i in self.input_reserve_filter_ids], axis=0)
            self.output_weight = tf.concat([self.output_weight, tf.truncated_normal((len(self.input_prune_filter_ids), self.output_dim), stddev=0.01, dtype=tf.float64)], axis=0)

            #prune and expand along the output dimension
            self.output_weight = tf.stack([self.output_weight[:, i] for i in self.output_reserve_filter_ids], axis=1)
            self.output_weight = tf.Variable(tf.concat([self.output_weight, tf.truncated_normal((self.input_dim + self.neigh_dim, len(self.output_prune_filter_ids)), stddev=0.01, dtype=tf.float64)], axis=1), trainable=True, name=self.name + 'output_weight')

            #gradient mask
            self.mask = np.concatenate([np.zeros((len(self.input_reserve_filter_ids), len(self.output_reserve_filter_ids))), np.ones((len(self.input_prune_filter_ids), len(self.output_reserve_filter_ids)))], axis=0)
            self.mask = np.concatenate([self.mask, np.ones((self.input_dim + self.neigh_dim, len(self.output_prune_filter_ids)))], axis=1)

        else:
            self.output_reserve_filter_ids = list(filter(lambda x: x not in self.output_prune_filter_ids, range(self.output_dim)))

            #prune and expand along the output dimension only
            self.output_weight = tf.stack([self.output_weight[:, i] for i in self.output_reserve_filter_ids], axis=1)
            self.output_weight = tf.Variable(tf.concat([self.output_weight, tf.truncated_normal((self.input_dim + self.neigh_dim, len(self.output_prune_filter_ids)), stddev=0.01, dtype=tf.float64)], axis=1), trainable=True, name=self.name + 'output_weight')

            #gradient mask
            self.mask = np.concatenate([np.zeros((self.input_dim + self.neigh_dim, len(self.output_reserve_filter_ids))), np.ones((self.input_dim + self.neigh_dim, len(self.output_prune_filter_ids)))], axis=1)

    def expand(self):
        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
        if self.input_prune:
            #self.input_prune_filter_ids = self.input_prune_filter_ids + self.input_dim
            #self.input_reserve_filter_ids = list(filter(lambda x: x not in self.input_prune_filter_ids, range(self.input_dim + self.neigh_dim)))
            #self.output_reserve_filter_ids = list(filter(lambda x: x not in self.output_prune_filter_ids, range(self.output_dim)))

            #prune and expand along the input dimension
            #self.output_weight = tf.stack([self.output_weight[i, :] for i in self.input_reserve_filter_ids], axis=0)

            #self.output_weight = tf.concat([self.output_weight, tf.truncated_normal((len(self.input_prune_filter_ids), self.output_weight.shape[1]), stddev=0.01, dtype=tf.float64)], axis=0)

            #self.output_weight = tf.concat([self.output_weight, tf.truncated_normal((len(self.input_prune_filter_ids), self.output_weight.shape[1]), stddev=0.01, dtype=tf.float64)], axis=0)
            #self.output_weight = tf.concat([self.output_weight, initializer([len(self.input_prune_filter_ids), self.output_weight.shape[1]])], axis=0)
           # self.output_weight = tf.concat([self.output_weight, initializer([len(self.input_prune_filter_ids), self.output_weight.shape[1]])], axis=0)

            #prune and expand along the output dimension
            #self.output_weight = tf.stack([self.output_weight[:, i] for i in self.output_reserve_filter_ids], axis=1)
            #self.output_weight = tf.Variable(tf.concat([self.output_weight, tf.truncated_normal((self.output_weight.shape[0], len(self.output_prune_filter_ids)), stddev=0.01, dtype=tf.float64)], axis=1), trainable=True, name=self.name + 'output_weight')
            #self.output_weight = tf.Variable(tf.concat([self.output_weight, tf.truncated_normal((self.output_weight.shape[0], len(self.output_prune_filter_ids)), stddev=0.01, dtype=tf.float64)], axis=1), trainable=True, name=self.name + 'output_weight')  
            #self.output_weight = tf.Variable(tf.concat([self.output_weight, initializer([self.output_weight.shape[0], len(self.output_prune_filter_ids)])], axis=1), trainable=True, name=self.name + 'output_weight')  
            self.output_weight = tf.Variable(initializer([self.output_weight.shape[0]+len(self.input_prune_filter_ids), self.output_weight.shape[1]+len(self.output_prune_filter_ids)]), trainable=True, name=self.name + 'output_weight')
            #gradient mask
            self.mask = None
            #self.mask = np.concatenate([np.zeros((len(self.input_reserve_filter_ids), len(self.output_reserve_filter_ids))), np.ones((len(self.input_prune_filter_ids), len(self.output_reserve_filter_ids)))], axis=0)
            #self.mask = np.concatenate([self.mask, np.ones((self.input_dim + self.neigh_dim, len(self.output_prune_filter_ids)))], axis=1)

        else:
            #self.output_reserve_filter_ids = list(filter(lambda x: x not in self.output_prune_filter_ids, range(self.output_dim)))

            #prune and expand along the output dimension only
            #self.output_weight = tf.stack([self.output_weight[:, i] for i in self.output_reserve_filter_ids], axis=1)
            #self.output_weight = tf.Variable(tf.concat([self.output_weight, initializer([self.output_weight.shape[0], len(self.output_prune_filter_ids)])], axis=1), trainable=True, name=self.name + 'output_weight')
            self.output_weight = tf.Variable(initializer([self.output_weight.shape[0], self.output_weight.shape[1]+len(self.output_prune_filter_ids)]), trainable=True, name=self.name + 'output_weight')
            #gradient mask
            self.mask = None
            #self.mask = np.concatenate([np.zeros((self.input_dim + self.neigh_dim, len(self.output_reserve_filter_ids))), np.ones((self.input_dim + self.neigh_dim, len(self.output_prune_filter_ids)))], axis=1)



    def __call__(self, self_matrix, neigh_matrix):
        batch_size = tf.shape(self_matrix)[0]
        neigh_matrix = tf.reshape(neigh_matrix, [batch_size, self.num_sample, self.neigh_dim])
        neigh_means = tf.reduce_mean(tf.cast(tf.to_float(neigh_matrix), tf.float64), axis=1) # batch_size * embed_size

        if self.neigh_dropout != 0:
            neigh_means = tf.nn.dropout(neigh_means, rate=self.neigh_dropout)

        output = tf.concat([self_matrix, neigh_means], axis=1)  # [N, input_dim+neigh_dim]
        output = self.act(tf.matmul(output, self.output_weight)) #[N, output_dim]
        return output


class MaxPoolAggregator(object):
    def __init__(self, name, num_sample, input_dim, neigh_dim, output_dim, activation, dropout=0.2):
        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

        self.neigh_weight = tf.Variable(initializer([neigh_dim * 2, output_dim // 2]), trainable=True, name=name + 'neigh_weight')
        self.weight = tf.Variable(initializer([input_dim, output_dim // 2]), trainable=True, name=name + 'self_weight')
        self.output_dim = output_dim
        self.act = activation
        self.neigh_dropout = dropout
        self.num_sample = num_sample
        self.input_dim = input_dim
        self.neigh_dim = neigh_dim

        self.mlp_layers = []
        self.mlp_layers.append(Dense(name=name,
                                     input_dim=neigh_dim,
                                     output_dim=neigh_dim * 2,
                                     act=self.act,
                                     dropout=dropout))
        self.mlp_layers.append(Dense(name=name + '_2',
                                     input_dim=neigh_dim * 2,
                                     output_dim=neigh_dim * 2,
                                     act=self.act,
                                     dropout=dropout))

    def __call__(self, self_matrix, neigh_matrix):
        batch_size = tf.shape(self_matrix)[0]
        neigh_matrix = tf.reshape(neigh_matrix, [batch_size * self.num_sample, self.neigh_dim])

        for dense_layer in self.mlp_layers:
            neigh_matrix = dense_layer(neigh_matrix)

        neigh_h = tf.reshape(neigh_matrix, [batch_size, self.num_sample, self.neigh_dim * 2])
        neigh_h = tf.reduce_max(neigh_h, axis=1) #[N, neigh_dim*2]

        from_neighs = tf.matmul(neigh_h, self.neigh_weight)
        from_self = tf.matmul(self_matrix, self.weight)

        output = tf.concat([from_self, from_neighs], axis=1)

        return self.act(output)

class MyMeanAggregator(object):
    ''' one weight with dimension (input_dim+neigh_dim, output_dim) on the concatenated vectors of
    self and neighbors after mean '''

    def __init__(self, name, num_sample, input_dim, neigh_dim, output_dim, activation, dropout=0.2):
        '''
        GCN mean aggregation for neighbor information
        :param name: name of defined aggregator
        :param num_sample: the number of positive neighbors user_user or item_item_graph
        :param input_dim: self_embedding dimension
        :param neigh_dim: neighbor_embedding dimension
        :param output_dim: output embedding dimension
        :param activation: activation function
        :param dropout: dropout rate
        '''

        #initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

        #self.output_weight = tf.Variable(initializer([input_dim + neigh_dim, output_dim]),
        #                                 name=name + 'output_weight')


        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

        self.output_weight = tf.Variable(initializer([input_dim + neigh_dim, output_dim]),
                                        name=name + 'output_weight')



        self.output_dim = output_dim
        self.input_dim = input_dim
        self.act = activation
        self.neigh_dropout = dropout
        self.num_sample = num_sample
        self.neigh_dim = neigh_dim

    def __call__(self, self_matrix, neigh_matrix):
        batch_size = tf.shape(self_matrix)[0]
        neigh_matrix = tf.reshape(neigh_matrix, [batch_size, self.num_sample, self.neigh_dim])
        neigh_means = tf.reduce_mean(tf.cast(tf.to_float(neigh_matrix), tf.float64), axis=1) # batch_size * embed_size

        if self.neigh_dropout != 0:
            neigh_means = tf.nn.dropout(neigh_means, rate=self.neigh_dropout)

        output = tf.concat([self_matrix, neigh_means], axis=1)  # [N, input_dim+neigh_dim]
        output = self.act(tf.matmul(output, self.output_weight)) #[N, output_dim]
        return output


class DEGC_MyMeanAggregator(object):
    ''' one weight with dimension (input_dim+neigh_dim, output_dim) on the concatenated vectors of
    self and neighbors after mean '''

    def __init__(self, name, num_sample, input_dim, neigh_dim, output_dim, activation, dropout=0.2):

        self.name = name
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.act = activation
        self.neigh_dropout = dropout
        self.num_sample = num_sample
        self.neigh_dim = neigh_dim
        self.initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)


    def load_conv_from_loadcheckpoint(self, load_checkpoint):
        self.output_weight = tf.Variable(initial_value=tf.train.load_variable(load_checkpoint, self.name+'output_weight'), trainable = False, name = self.name + 'output_weight', dtype=tf.float64)
    
    def load_conv_from_selected(self, selected_params):
        self.output_weight = tf.get_variable(name = self.name + 'output_weight', dtype=tf.float64, initializer = selected_params, trainable = True)

    def init_conv(self):
        self.output_weight = tf.Variable(self.initializer([self.input_dim + self.neigh_dim, self.output_dim]), trainable=True,
                                            name=self.name + 'output_weight')

    def extend_conv(self, ex_k):
        self.ex_k = ex_k
        with tf.compat.v1.variable_scope('model', reuse=True):
            previous_weight = tf.get_variable(self.name + 'output_weight', trainable = False)
            prev_dim = previous_weight.get_shape().as_list()[0]
            next_dim = previous_weight.get_shape().as_list()[1]
            # bottom to top
            new_weight_b2t = tf.get_variable(self.name + 'new_b2t' + 'output_weight', initializer=self.initializer([prev_dim, self.ex_k]),trainable=True)
            expanded_weight = tf.concat([previous_weight, new_weight_b2t], 1)
            # top to bottom
            new_weight_t2b = tf.get_variable(self.name + 'new_t2b' + 'output_weight', initializer=self.initializer([self.ex_k, next_dim + self.ex_k]),trainable=True)
            self.output_weight = tf.concat([expanded_weight, new_weight_t2b], 0)

    def extend_top_conv(self, ex_top_k):
        self.ex_top_k = ex_top_k
        with tf.compat.v1.variable_scope('model', reuse=True):
            previous_weight = tf.get_variable(self.name + 'output_weight', trainable = False)
            new_weight = tf.get_variable(self.name + 'new_top' + 'output_weight', initializer=self.initializer([self.ex_top_k, self.output_dim]),trainable=True)
            self.output_weight = tf.concat([previous_weight, new_weight], 0)

    def extend_bottom_conv(self, ex_bottom_k):
        self.ex_bottom_k = ex_bottom_k
        with tf.compat.v1.variable_scope('model', reuse=True):
            previous_weight = tf.get_variable(self.name + 'output_weight', trainable = False)
            prev_dim = previous_weight.get_shape().as_list()[0]
            new_weight = tf.get_variable(self.name + 'new_bottom' + 'output_weight', initializer=self.initializer([prev_dim, self.ex_bottom_k]), trainable = True)
            self.output_weight = tf.concat([previous_weight, new_weight], 1)
        
    def extend_homogeneous_conv(self, ex_homogeneous_k):
        self.ex_homogeneous_k = ex_homogeneous_k
        with tf.compat.v1.variable_scope('model', reuse=True):
            self.output_weight = tf.get_variable(self.name + 'output_weight', trainable = False)

    def load_predict_conv(self):
        with tf.compat.v1.variable_scope('model', reuse=True):
            self.output_weight = tf.get_variable(self.name + 'output_weight', trainable = True)


    def __call__(self, self_matrix, neigh_matrix):
        batch_size = tf.shape(self_matrix)[0]
        neigh_matrix = tf.reshape(neigh_matrix, [batch_size, self.num_sample, self.neigh_dim])
        neigh_means = tf.reduce_mean(tf.cast(tf.to_float(neigh_matrix), tf.float64), axis=1) # batch_size * embed_size

        if self.neigh_dropout != 0:
            neigh_means = tf.nn.dropout(neigh_means, rate=self.neigh_dropout)

        output = tf.concat([self_matrix, neigh_means], axis=1)  # [N, input_dim+neigh_dim]
        output = self.act(tf.matmul(output, self.output_weight)) #[N, output_dim]
        return output