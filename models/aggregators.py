import tensorflow as tf
from .layers import Dense


class MyMeanAggregator(object):
    ''' one weight with dimension (input_dim+neigh_dim, output_dim) on the concatenated vectors of
    self and neighbors after mean '''

    def __init__(self, load_checkpoint, name, num_sample, input_dim, neigh_dim, output_dim, activation, dropout=0.2):
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

        if load_checkpoint == '':
            initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

            self.output_weight = tf.Variable(initializer([input_dim + neigh_dim, output_dim]),
                                            name=name + 'output_weight')
        else:
            print('load_checkpoint', load_checkpoint)
            print('initial value', tf.train.load_variable(load_checkpoint, name+'output_weight'))
            self.output_weight = tf.Variable(initial_value=tf.train.load_variable(load_checkpoint, name+'output_weight'), name = name + 'output_weight', dtype=tf.float64)


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


class MaxPoolAggregator(object):
    def __init__(self, name, num_sample, input_dim, neigh_dim, output_dim, activation, dropout=0.2):
        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

        self.neigh_weight = tf.Variable(initializer([neigh_dim * 2, output_dim // 2]), name=name + 'neigh_weight')
        self.weight = tf.Variable(initializer([input_dim, output_dim // 2]), name=name + 'self_weight')
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

"""
class LightGCNAggregator(object):
    ''' one weight with dimension (neigh_dim, output_dim) on the neighbors after mean, without nonlinear activation '''

    def __init__(self, name, num_sample, neigh_dim, output_dim, batch_size, dropout=0.2):
        '''
        GCN mean aggregation for neighbor information
        :param name: name of defined aggregator
        :param num_sample: the number of positive neighbors user_user or item_item_graph
        :param neigh_dim: neighbor_embedding dimension
        :param output_dim: output embedding dimension
        :param batch_size: batch_size
        :param dropout: dropout rate
        '''

        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

        self.output_weight = tf.Variable(initializer([neigh_dim, output_dim]),
                                         name=name + 'output_weight')
        self.output_dim = output_dim
        self.neigh_dropout = dropout
        self.batch_size = batch_size
        self.num_sample = num_sample
        self.neigh_dim = neigh_dim

    def __call__(self, neigh_matrix):
        neigh_matrix = tf.reshape(neigh_matrix, [self.batch_size, self.num_sample, self.neigh_dim])
        neigh_means = tf.reduce_mean(tf.cast(tf.to_float(neigh_matrix), tf.float64), axis=1) # batch_size * embed_size

        if self.neigh_dropout != 0:
            neigh_means = tf.nn.dropout(neigh_means, rate=self.neigh_dropout)

        output = neigh_means  # [N, neigh_dim]
        output = tf.matmul(output, self.output_weight) #[N, output_dim]
        return output
"""

class LightGCNAggregator(object):
    ''' one weight with dimension (neigh_dim, output_dim) on the neighbors after mean, without nonlinear activation '''

    def __init__(self, load_checkpoint, name, num_sample, input_dim, neigh_dim, output_dim, dropout=0.2):
        '''
        GCN mean aggregation for neighbor information
        :param name: name of defined aggregator
        :param num_sample: the number of positive neighbors user_user or item_item_graph
        :param neigh_dim: neighbor_embedding dimension
        :param output_dim: output embedding dimension
        :param batch_size: batch_size
        :param dropout: dropout rate
        '''

        if load_checkpoint == '':
            initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

            self.output_weight = tf.Variable(initializer([input_dim + neigh_dim, output_dim]),
                                            name=name + 'output_weight')
        else:
            print('load_checkpoint', load_checkpoint)
            self.output_weight = tf.Variable(initial_value=tf.train.load_variable(load_checkpoint, name+'output_weight'), name = name + 'output_weight', dtype=tf.float64)
            
        self.output_dim = output_dim
        self.neigh_dropout = dropout
        self.num_sample = num_sample
        self.neigh_dim = neigh_dim

    def __call__(self, self_matrix, neigh_matrix):
        batch_size = tf.shape(self_matrix)[0]
        neigh_matrix = tf.reshape(neigh_matrix, [batch_size, self.num_sample, self.neigh_dim])
        neigh_means = tf.reduce_mean(tf.cast(tf.to_float(neigh_matrix), tf.float64), axis=1) # batch_size * embed_size

        if self.neigh_dropout != 0:
            neigh_means = tf.nn.dropout(neigh_means, rate=self.neigh_dropout)

        #output = neigh_means  # [N, neigh_dim]
        output = tf.concat([self_matrix, neigh_means], axis=1) # [N, input_dim + neigh_dim]
        output = tf.matmul(output, self.output_weight) #[N, output_dim]
        return output

class LightGCNAggregator_Raw(object):
    ''' one weight with dimension (neigh_dim, output_dim) on the neighbors after mean, without nonlinear activation '''

    def __init__(self, load_checkpoint, name, num_sample, input_dim, neigh_dim, output_dim, dropout=0.2):
        '''
        GCN mean aggregation for neighbor information
        :param name: name of defined aggregator
        :param num_sample: the number of positive neighbors user_user or item_item_graph
        :param neigh_dim: neighbor_embedding dimension
        :param output_dim: output embedding dimension
        :param batch_size: batch_size
        :param dropout: dropout rate
        '''


        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

        self.output_weight = tf.Variable(initializer([input_dim + neigh_dim, output_dim]),
                                            name=name + 'output_weight')

        self.output_dim = output_dim
        self.neigh_dropout = dropout
        self.num_sample = num_sample
        self.neigh_dim = neigh_dim

    def __call__(self, self_matrix, neigh_matrix):
        batch_size = tf.shape(self_matrix)[0]
        neigh_matrix = tf.reshape(neigh_matrix, [batch_size, self.num_sample, self.neigh_dim])
        neigh_means = tf.reduce_mean(tf.cast(tf.to_float(neigh_matrix), tf.float64), axis=1) # batch_size * embed_size

        if self.neigh_dropout != 0:
            neigh_means = tf.nn.dropout(neigh_means, rate=self.neigh_dropout)

        #output = neigh_means  # [N, neigh_dim]
        output = tf.concat([self_matrix, neigh_means], axis=1) # [N, input_dim + neigh_dim]
        output = tf.matmul(output, self.output_weight) #[N, output_dim]
        return output

class MyMeanAggregator_Raw(object):
    ''' one weight with dimension (input_dim+neigh_dim, output_dim) on the concatenated vectors of
    self and neighbors after mean '''

    def __init__(self, load_checkpoint, name, num_sample, input_dim, neigh_dim, output_dim, activation, dropout=0.2):
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



