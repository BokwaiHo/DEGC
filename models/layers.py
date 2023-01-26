import tensorflow as tf


class FCL(object):
    """ Dense layer (from GraphSage TensorFlor at Github) """
    def __init__(self, name, input_dim, output_dim, dropout=0., act=tf.nn.tanh):
        '''
        Dense layers
        :param name: name of layer
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param dropout: dropout ratio
        :param act: activation function
        '''
        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
        self.dropout = dropout

        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense_weight = tf.Variable(initializer([input_dim, output_dim]), name=name)

    def __call__(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        output = tf.matmul(x, self.dense_weight)

        return self.act(output)

class Dense(object):
    """ Dense layer (from GraphSage TensorFlor at Github) """
    def __init__(self, name, input_dim, output_dim, dropout=0., act=tf.nn.tanh):
        '''
        Dense layers
        :param name: name of layer
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param dropout: dropout ratio
        :param act: activation function
        '''
        self.dropout = dropout

        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim

        with tf.variable_scope(name + 'dense'):
            self.dense_weight = tf.get_variable('weights', shape=(input_dim, output_dim),
                                                dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        output = tf.matmul(x, self.dense_weight)

        return self.act(output)
