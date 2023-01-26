import time
import numpy as np
import tensorflow as tf

class BPRMF(object):
    def __init__(self, dataset_argv, architect_argv, adj_degree, 
                 num_samples, ptmzr_argv, act,
                 neigh_drop_rate, l2_embed,
                 neg_item_num=5, inc_reg=[0,0],
                 old_num_user=0, old_num_item=0,
                 distill_mode=""):

        self.graph = tf.Graph()
        start = time.time()

        with self.graph.as_default():
            (self.num_user, self.num_item) = dataset_argv
            self.old_num_user, self.old_num_item = old_num_user, old_num_item

            self.layer_dims = architect_argv
            self.neg_item_num = neg_item_num

            self.u_id = tf.placeholder(tf.int64, shape=[None, ], name='u_id')
            self.pos_item_id = tf.placeholder(tf.int64, shape=[None, ], name='pos_item_id')
            self.neg_item_id = tf.placeholder(tf.int64, shape=[None, neg_item_num], name='neg_item_id')

            # ===
            self.num_samples = num_samples
            self.l2_embed = l2_embed
            gcn_act = eval(act)

            initializer = tf.contrib.layers.xavier_initializer(dtype=tf.dtypes.float64)
            self.user_embeddings = tf.Variable(initializer([self.num_user, self.layer_dims[0]]), name='user_embedding')
            self.item_embeddings = tf.Variable(initializer([self.num_item, self.layer_dims[0]]), name='item_embedding')

            # ===
            self.bpr_loss = self.model_fn('model', inc_reg=inc_reg)
            _learning_rate, _epsilon = ptmzr_argv[1:3]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                assert ptmzr_argv[0].lower() == 'adam'
                # self.ptmzr = tf.train.AdamOptimizer(learning_rate=_learning_rate,
                #                                     epsilon=_epsilon).minimize(self.bpr_loss)
                self.ptmzr = tf.train.RMSPropOptimizer(learning_rate=_learning_rate).minimize(self.bpr_loss)
            print('gnn end: ', time.time() - start)

    def model_fn(self, scope, inc_reg=[0,0]):
        with tf.variable_scope(scope):
            # self.user_embeddings = tf.get_variable('user_embedding', dtype=tf.float64,
            #                                        shape=[self.num_user, self.layer_dims[0]])
            # self.item_embeddings = tf.get_variable('item_embedding', dtype=tf.float64,
            #                                        shape=[self.num_item, self.layer_dims[0]])        

            user_rep = tf.nn.embedding_lookup(self.user_embeddings, self.u_id)
            pos_item_rep = tf.nn.embedding_lookup(self.item_embeddings, self.pos_item_id)
            neg_item_rep = tf.nn.embedding_lookup(self.item_embeddings, self.neg_item_id)
            item_rep = tf.concat([tf.expand_dims(pos_item_rep, 1), neg_item_rep], 1)

            # === BPR loss
            pos_rating = tf.reduce_sum(tf.multiply(user_rep, pos_item_rep), 1)
            pos_rating = tf.expand_dims(pos_rating, 1)
            pos_rating = tf.tile(pos_rating, [1, self.neg_item_num])
            pos_rating = tf.reshape(pos_rating, [tf.shape(self.u_id)[0] * self.neg_item_num, 1])

            
            user_rep = tf.tile(user_rep, [1, self.neg_item_num])
            user_rep = tf.reshape(user_rep, [tf.shape(user_rep)[0], self.neg_item_num, 64])
            neg_rating = tf.reduce_sum(tf.multiply(user_rep, neg_item_rep), 2)
            
            # batch_neg_item_embedding = tf.transpose(neg_item_rep, [0, 2, 1])
            # neg_rating = tf.matmul(tf.expand_dims(user_rep, 1), batch_neg_item_embedding)
            # neg_rating = tf.squeeze(neg_rating, 1)
            neg_rating = tf.reshape(neg_rating, tf.shape(pos_rating))

            bpr_loss = pos_rating - neg_rating
            bpr_loss = tf.nn.sigmoid(bpr_loss)
            # bpr_loss = tf.negative(tf.log(bpr_loss))
            # bpr_loss = tf.reduce_mean(bpr_loss)
            bpr_loss = tf.negative(tf.reduce_sum(tf.log(bpr_loss)))
            # bpr_loss = tf.reduce_mean(bpr_loss)

            self.user_reg = bpr_loss
            self.l2_reg = self.l2_embed * (tf.nn.l2_loss(user_rep) + tf.nn.l2_loss(item_rep))
            bpr_loss += self.l2_reg

            self.user_reg = tf.constant(0.0, dtype=tf.float64)
            self.item_reg = tf.constant(0.0, dtype=tf.float64)
        return bpr_loss

    def predict(self, batch_user_idx, item_idx, test_n_user):

        # === process item rep
        # # get item_rep for existing items in the training set
        n_item = len(item_idx)
        item_idx = item_idx[:self.num_item]
        item_idx = tf.convert_to_tensor(item_idx)
        item_idx = tf.cast(item_idx, tf.int32)

        item_rep = tf.nn.embedding_lookup(self.item_embeddings, item_idx)
        # # random init item_rep for previously unseen items
        if n_item > self.num_item:
            try:
                new_item_rep = tf.get_default_graph().get_tensor_by_name("new_item_rep:0")
            except KeyError:
                new_item_rep = tf.random.uniform([n_item-self.num_item, self.layer_dims[2]], dtype=tf.float64, name='new_item_rep')
            item_rep = tf.concat([item_rep, new_item_rep], axis=0)

        # === process user rep
        # # random init user for previously unseen users
        if test_n_user > self.num_user:
            try:
                new_user_rep = tf.get_default_graph().get_tensor_by_name("new_user_rep:0")
            except KeyError:
                new_user_rep = tf.random.uniform([test_n_user-self.num_user, self.layer_dims[2]], dtype=tf.float64, name='new_user_rep')
        # # process mini-batches for users accordingly
        if batch_user_idx[0] >= self.num_user:
            user_rep = new_user_rep[batch_user_idx[0]-self.num_user:batch_user_idx[-1]-self.num_user+1]
        elif batch_user_idx[0] < self.num_user and self.num_user <= batch_user_idx[-1]:
            n_user_batch = len(batch_user_idx)
            batch_user_idx = batch_user_idx[batch_user_idx < self.num_user]
            batch_user_idx = tf.convert_to_tensor(batch_user_idx)
            batch_user_idx = tf.cast(batch_user_idx, tf.int32)
            user_rep = tf.nn.embedding_lookup(self.user_embeddings, batch_user_idx)
            user_rep = tf.concat([user_rep, new_user_rep[:n_user_batch-user_rep.shape[0].value]], axis=0)
        else:
            batch_user_idx = tf.convert_to_tensor(batch_user_idx)
            batch_user_idx = tf.cast(batch_user_idx, tf.int32)
            user_rep = tf.nn.embedding_lookup(self.user_embeddings, batch_user_idx)

        rating_score = tf.matmul(user_rep, item_rep, transpose_b=True)

        return rating_score, user_rep, item_rep

