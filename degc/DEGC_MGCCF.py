from cmath import phase
from re import X
from sqlite3 import paramstyle
from xxlimited import new
import tensorflow as tf
import sys
from models.layers import Dense, FCL
from degc.DEGC_aggregators import MyMeanAggregator, DEGC_MyMeanAggregator
import numpy as np
import random
from data_utils.utils import *

"""
TensorFlow Implementation for MGCCF with tunable width
__author__: Bowei He
__date__: Aug. 2022
__copyright__: Hong Kong Research Center, Huawei Technologies
"""
def calculate_contrastive_loss(old_embedding, old_num, cur_pos_neigh, cur_neg_neigh, tau, selected_id):
    '''

    :param old_embedding: The embedding from previous time point
    :param old_num: The number of users/items of previous time point
    :param cur_pos_neigh: The embedding of time point t of positive neighbors from previous time point
    :param cur_neg_neigh: The embedding of time point t of negative neighbors from previous time point
    :param tau: hyperparameter
    :param user_weight: adaptive distillation weights for each user
    :return: contrastive loss term
    '''

    old_emb = tf.nn.embedding_lookup(old_embedding[:old_num], selected_id)
    selected_pos_neigh = tf.nn.embedding_lookup(cur_pos_neigh, selected_id)
    selected_neg_neigh = tf.nn.embedding_lookup(cur_neg_neigh, selected_id)
    numerator_user = tf.math.exp(
        tf.reduce_sum(tf.multiply(tf.expand_dims(old_emb, 1),
                                  selected_pos_neigh) / tau, axis=2))
    denom_user = tf.reduce_sum(tf.math.exp(
        tf.reduce_sum(tf.multiply(tf.expand_dims(old_emb, 1),
                                  selected_neg_neigh) / tau, axis=2)))

 
    ct_loss = tf.reduce_mean(tf.reduce_mean(-tf.math.log(numerator_user / denom_user), 1))
    return ct_loss

def uniform_sample(ids, adj, num_samples=15):
    ''' uniform sample for central nodes
    :param ids: (N, )
    :param num_samples: number of sampled neighbors
    :return: adj_list with feature and ids. shape: (none, num_samples)
    '''

    adj_lists = tf.nn.embedding_lookup(adj, ids)  # (N, 128)
    adj_lists = tf.transpose(tf.random.shuffle(tf.transpose(adj_lists)))
    adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])  # (N, num_samples)
    return adj_lists

class DEGC_MGCCF(object):
    def __init__(self, segment_id, new_node_init, train_matrix, user_node_info, item_node_info, load_checkpoint, dataset_argv, architect_argv, adj_degree, num_samples,
                 ptmzr_argv, aggregator, act, neigh_drop_rate, l2_embed, dist_embed, num_self_neigh=10,
                 neg_item_num=10, pretrain_data=True, num_neigh=15, l1_lambda=0.001, l2_lambda=0.01, gl_lambda=0.01, ex_k=64):

        self.segment_id = segment_id
        self.aggregator = aggregator
        self.ptmzr_argv = ptmzr_argv
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.gl_lambda = gl_lambda
        self.ex_k = ex_k
        self.neigh_drop_rate = neigh_drop_rate
        self.adj_degree = adj_degree
        self.graph = tf.Graph()


        #with self.graph.as_default():
        (self.input_dim, self.num_user, self.num_item) = dataset_argv
        self.new_node_init = new_node_init
        self.train_matrix = train_matrix
        self.n_user_train, self.n_old_user = user_node_info
        self.n_item_train, self.n_old_item = item_node_info
        self.load_checkpoint = load_checkpoint
        self.layer_dims = architect_argv
        self.neg_item_num = neg_item_num
        print('input dim: %d\n'
                'neigh_drop_rate: %g\nl2(lambda): %g\n' %
                (self.input_dim, neigh_drop_rate, l2_embed))
        #self.num_neigh = tf.cast(num_neigh, dtype=tf.int32)
        self.num_neigh = num_neigh
        self.layer_dims = [self.input_dim] + self.layer_dims
        self.num_layers = len(self.layer_dims)  # Total number of NN layers
        self.num_self_neigh = num_self_neigh
        self.gcn_act = eval(act)

        # ===============================

        self.num_samples = num_samples
        self.l2_embed = l2_embed
        self.dist_embed = dist_embed
        self.pretrain_data = pretrain_data
        self.neigh_dropout = neigh_drop_rate           
        self.graph_conv_params = dict()
                
            # Begin DEGC algorithm





    def apply_prune_on_grads(self, grads_and_vars, total_mask):
        for i in range(0, 2 * len(total_mask), 4):
            #user graphconv
            grads_and_vars[i+1] = (tf.multiply(grads_and_vars[i+1][0], total_mask[i//2]),grads_and_vars[i+1][1])
            #item graphconv        
            grads_and_vars[i+3] = (tf.multiply(grads_and_vars[i+3][0], total_mask[i//2 + 1]),grads_and_vars[i+3][1])
        return grads_and_vars

    
    def model_fn(self, scope, final):
        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
        with tf.compat.v1.variable_scope(scope, reuse=True):
            if not self.pretrain_data :
                self.user_embeddings = tf.Variable(initializer([self.num_user, self.input_dim]), trainable=True, name='user_embedding')
                self.item_embeddings = tf.Variable(initializer([self.num_item, self.input_dim]), trainable=True, name='item_embedding')
                print('=== using xavier initialization for embeddings, no pretrain')
            else:
                if final:
                    embedding_train = True
                else:
                    embedding_train = False
                self.user_embeddings = tf.Variable(initializer([self.num_user, self.input_dim]), trainable=embedding_train, name='user_embedding')
                self.item_embeddings = tf.Variable(initializer([self.num_item, self.input_dim]), trainable=embedding_train, name='item_embedding')

                old_u_emb_val = tf.train.load_variable(self.load_checkpoint, 'model/user_embedding')
                old_i_emb_val = tf.train.load_variable(self.load_checkpoint, 'model/item_embedding')
                
                #update_1 = tf.assign(self.user_embeddings[:old_u_emb_val.shape[0], ], old_u_emb_val)
                #update_2 = tf.assign(self.item_embeddings[:old_i_emb_val.shape[0], ], old_i_emb_val)

                self.user_embeddings[:old_u_emb_val.shape[0], ].assign(old_u_emb_val)
                self.item_embeddings[:old_i_emb_val.shape[0], ].assign(old_i_emb_val)

                if self.new_node_init == '2hop_mean':
                    if self.n_user_train > self.n_old_user:
                        new_users_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('user', old_u_emb_val, self.train_matrix)
                        #update_3 = tf.assign(self.user_embeddings[self.n_old_user:,], new_users_init)
                        self.user_embeddings[self.n_old_user:,].assign(new_users_init)

                    if self.n_item_train > self.n_old_item:
                        new_items_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('item', old_i_emb_val, self.train_matrix)
                        #update_4 = tf.assign(self.item_embeddings[self.n_old_item:,], new_items_init)
                        self.item_embeddings[self.n_old_item:,].assign(new_items_init)

                print('=== using pretrained data for initializing embeddings, but still trainable')
            #self.user_medium_input_1 = tf.Variable(initializer([self.num_item, self.layer_dims[1]]), name='user_medium_input_1')
            #self.item_medium_input_1 = tf.Variable(initializer([self.num_user, self.layer_dims[1]]), name='item_medium_input_1')
            #self.user_medium_input_2 = tf.Variable(initializer([self.num_user, self.layer_dims[2]]), name='user_medium_input_2')
            #self.item_medium_input_2 = tf.Variable(initializer([self.num_item, self.layer_dims[2]]), name='item_medium_input_2')


            batch_size = tf.shape(self.u_id)[0]

            pos_users = tf.reshape(uniform_sample(self.u_id, self.u_u_graph_ph, 1), [-1, ])
            #all_user_rep, all_user_embed, self.user_medium_input_1, self.user_medium_input_2 = self.graphconv('user_gcn', tf.concat([self.u_id, pos_users], 0),
            #                                              self.user_embeddings, self.item_embeddings, 'user', True)

            all_user_rep, all_user_embed = self.graphconv('user_gcn', tf.concat([self.u_id, pos_users], 0),
                                                          self.user_embeddings, self.item_embeddings, 'user', False)

            all_user_rep = tf.concat([all_user_rep, all_user_embed], 1)

            user_rep, pos_user_rep = tf.split(all_user_rep, [batch_size, batch_size])
            user_embed, pos_user_embed = tf.split(all_user_embed, [batch_size, batch_size])

            user_user_distance = tf.reduce_sum(tf.math.pow(user_rep - pos_user_rep, 2)) \
                                 + tf.reduce_sum(tf.math.pow(user_embed - pos_user_embed, 2))
            user_rep = tf.expand_dims(user_rep, 1)


            # ================================================================
            pos_items_neighs = tf.reshape(uniform_sample(self.pos_item_id, self.v_v_graph_ph, 1), [-1, ])
            #all_pos_item_rep, all_pos_item_embed, self.item_medium_input_1, self.item_medium_input_2 = self.graphconv('pos_item_gcn',
            #                                                      tf.concat([self.pos_item_id, pos_items_neighs], 0),
            #                                                      self.user_embeddings, self.item_embeddings,
            #                                                      'pos_item', True)

            all_pos_item_rep, all_pos_item_embed = self.graphconv('pos_item_gcn',
                                                                  tf.concat([self.pos_item_id, pos_items_neighs], 0),
                                                                  self.user_embeddings, self.item_embeddings,
                                                                  'pos_item', False)

            all_pos_item_rep = tf.concat([all_pos_item_rep, all_pos_item_embed], 1)
            pos_item_rep, pos_item_neigh_rep = tf.split(all_pos_item_rep, [batch_size, batch_size])
            pos_item_embed, pos_item_neigh_embed = tf.split(all_pos_item_embed, [batch_size, batch_size])

            pos_item_item_dist = tf.reduce_sum(tf.math.pow(pos_item_rep - pos_item_neigh_rep, 2)) \
                                 + tf.reduce_sum(tf.math.pow(pos_item_embed - pos_item_neigh_embed, 2))

            # ============================
            #neg_item_rep, neg_item_embed, self.item_medium_input_1, self.item_medium_input_2 = self.graphconv('neg_item_gcn', self.neg_item_id, self.user_embeddings,
            #                                              self.item_embeddings, 'neg_item', True)

            neg_item_rep, neg_item_embed = self.graphconv('neg_item_gcn', self.neg_item_id, self.user_embeddings,
                                                          self.item_embeddings, 'neg_item', False)

            neg_item_rep = tf.concat([neg_item_rep, neg_item_embed], 2)
            item_rep = tf.concat([tf.expand_dims(pos_item_rep, 1), neg_item_rep], 1)
            item_embed = tf.concat([tf.expand_dims(pos_item_embed, 1), neg_item_embed], 1)

            # self.item_medium_input = tf.concat([pos_item_layer_input, neg_item_layer_input], 1)

            # === BPR loss
            pos_rating = tf.reduce_sum(tf.multiply(tf.squeeze(user_rep, 1), pos_item_rep), 1)
            pos_rating = tf.expand_dims(pos_rating, 1)
            pos_rating = tf.tile(pos_rating, [1, self.neg_item_num])
            pos_rating = tf.reshape(pos_rating, [tf.shape(pos_rating)[0] * self.neg_item_num, 1])

            batch_neg_item_embedding = tf.transpose(neg_item_rep, [0, 2, 1])
            neg_rating = tf.matmul(user_rep, batch_neg_item_embedding)
            neg_rating = tf.squeeze(neg_rating, 1)
            neg_rating = tf.reshape(neg_rating, [tf.shape(neg_rating)[0] * self.neg_item_num, 1])

            bpr_loss = pos_rating - neg_rating
            bpr_loss = tf.nn.sigmoid(bpr_loss)
            bpr_loss = -tf.math.log(bpr_loss)
            bpr_loss = tf.reduce_sum(bpr_loss)

            reg_loss = self.l2_embed * (tf.nn.l2_loss(user_rep) + tf.nn.l2_loss(item_rep))

        return bpr_loss, reg_loss, self.dist_embed * (user_user_distance + pos_item_item_dist)


    def graphconv(self, scope, central_ids, user_embeddings, item_embeddings, tag, medium=False):
        with tf.compat.v1.variable_scope(scope, reuse=True):
            if tag == 'user':
                agg_funcs = self.user_agg_funcs
                self_agg_funcs = self.u_u_agg_func
                self_embeddings = user_embeddings
                neigh_embeddings = item_embeddings
                self_adj_info_ph = self.u_adj_info_ph
                neigh_adj_info_ph = self.v_adj_info_ph
                self_graph_info = self.u_u_graph_ph
                embed = tf.gather(self_embeddings, central_ids)
                if medium:
                    medium_input_1 = self.user_medium_input_1
                    medium_input_2 = self.user_medium_input_2 
            else:
                agg_funcs = self.item_agg_funcs
                self_agg_funcs = self.v_v_agg_func
                self_embeddings = item_embeddings
                neigh_embeddings = user_embeddings
                self_adj_info_ph  = self.v_adj_info_ph
                neigh_adj_info_ph = self.u_adj_info_ph
                self_graph_info = self.v_v_graph_ph
                embed = tf.gather(self_embeddings, central_ids)
                if medium:
                    medium_input_1 = self.item_medium_input_1
                    medium_input_2 = self.item_medium_input_2
                if tag != 'pos_item':
                    central_ids = tf.reshape(central_ids, [tf.shape(central_ids)[0] * central_ids.get_shape()[1]])

            central_ids = tf.cast(central_ids, tf.int32)
            unique_nodes, unique_idx = tf.unique(central_ids)

            self_id_at_layers = [unique_nodes]
            neigh_id_at_layers = []

            # == Bipartite GCN # =================================================================================

            for i in range(self.num_layers - 1):
                neigh_id_at_layer_i = uniform_sample(self_id_at_layers[i],
                                                     self_adj_info_ph if i % 2 == 0 else neigh_adj_info_ph,
                                                     self.num_samples[i])
                neigh_id_at_layers.append(neigh_id_at_layer_i)
                if i + 1 < self.num_layers - 1:
                    self_id_at_layers.append(tf.reshape(neigh_id_at_layers[i - 1], [-1]))

            self_matrix_at_layers = [tf.gather(self_embeddings if i % 2 == 0 else neigh_embeddings,
                                               self_id_at_layers[i]) for i in range(self.num_layers - 1)]

            neigh_matrix_at_layers = [tf.gather(neigh_embeddings if i % 2 == 0 else self_embeddings,
                                                neigh_id_at_layers[i]) for i in range(self.num_layers - 1)]

            for i in range(self.num_layers - 2, -1, -1):
                output1 = agg_funcs[i](self_matrix_at_layers[i], neigh_matrix_at_layers[i])

                if i > 0:
                    neigh_matrix_at_layers[i - 1] = tf.reshape(output1, [tf.shape(self_matrix_at_layers[i - 1])[0],
                                                                         self.num_samples[i - 1], -1])
                if medium:
                    medium_id, medium_idx = tf.unique(self_id_at_layers[i])
                    medium_id = tf.cast(medium_id, tf.int64)
                    medium_idx, _ = tf.unique(medium_idx)
                    medium_idx = tf.cast(medium_idx, tf.int64)
                    medium_input = tf.nn.embedding_lookup(output1, medium_idx)
                    if i == self.num_layers-2:
                        medium_original = tf.nn.embedding_lookup(medium_input_1, medium_id)
                        assigned_value = medium_input - medium_original
                        delta = tf.scatter_nd(tf.expand_dims(medium_id,1), assigned_value, tf.shape(medium_input_1, out_type=tf.int64))
                        medium_input_1 = medium_input_1 + delta

                    elif i == self.num_layers-3:
                        medium_original = tf.nn.embedding_lookup(medium_input_2, medium_id)
                        assigned_value = medium_input - medium_original
                        delta = tf.scatter_nd(tf.expand_dims(medium_id,1), assigned_value, tf.shape(medium_input_2, out_type=tf.int64))
                        medium_input_2 = medium_input_2 + delta



                # == MGE layer # =====================================================================================
                self_graph_neighs = uniform_sample(unique_nodes, self_graph_info, self.num_self_neigh)
                self_graph_neighs = tf.cast(self_graph_neighs, tf.int32)
                # self_graph_neighs = tf.nn.embedding_lookup(self.u_u_graph_ph, unique_nodes)
                self_graph_neighs_matrix = tf.gather(self_embeddings, self_graph_neighs)
                output2 = self_agg_funcs(self_matrix_at_layers[0], self_graph_neighs_matrix)
                # ====================================================================================================

                output1 = tf.nn.embedding_lookup(output1, unique_idx)
                output2 = tf.nn.embedding_lookup(output2, unique_idx)
                output = tf.concat([output1, output2], 1)
                output = tf.nn.tanh(output)
                if tag != "pos_item" and tag != "user":
                    output_shape = output.get_shape().as_list()
                    output = tf.reshape(output, [tf.shape(embed)[0], self.neg_item_num, output_shape[-1]])

            if not medium:
                return output, embed
            else:
                return output, embed, medium_input_1, medium_input_2

    
    def predict(self, batch_user_idx, item_idx, test_n_user, new_user_embedding=None, new_item_embedding=None):
        # ==process item rep
        # # get item_rep for existing items in the train set
        n_item = len(item_idx)
        item_idx = tf.convert_to_tensor(item_idx, dtype=tf.int32)

        # === process new user rep
        if test_n_user > self.num_user:
            assert new_user_embedding.shape[0] == test_n_user - self.num_user
            user_embed = tf.concat([self.user_embeddings, new_user_embedding], axis=0)
        else:
            user_embed = self.user_embeddings

        # === process 
        if n_item > self.num_item:
            assert new_item_embedding.shape[0] == n_item - self.num_item
            item_embed = tf.concat([self.item_embeddings, new_item_embedding], axis=0)
        else:
            item_embed = self.item_embeddings

        
        batch_user_idx = tf.convert_to_tensor(batch_user_idx)
        batch_user_idx = tf.cast(batch_user_idx, tf.int32)
        batch_user_rep, batch_user_embed = self.graphconv('user_gcn', batch_user_idx, user_embed, item_embed, 'user')
        batch_user_rep = tf.concat([batch_user_rep, batch_user_embed], 1)

        batch_item_rep, batch_item_embed = self.graphconv('item_gcn', item_idx, user_embed, item_embed, 'pos_item')
        batch_item_rep = tf.concat([batch_item_rep, batch_item_embed], 1)

        rating_score = tf.matmul(batch_user_rep, batch_item_rep, transpose_b=True)
        
        return rating_score, batch_user_rep, batch_item_rep

    def build_model(self, prediction=False, expansion=None):
        final = False
        tf.reset_default_graph()
        with self.graph.as_default():
            self.u_id = tf.placeholder(tf.int32, shape=[None, ], name='u_id')
            self.random_ids = tf.placeholder(tf.int32, shape=[None, ], name='random_id')
            self.pos_item_id = tf.placeholder(tf.int32, shape=[None, ], name='pos_item_id')
            self.neg_item_id = tf.placeholder(tf.int32, shape=[None, self.neg_item_num], name='neg_item_id')

            self.u_adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.adj_degree], name='u_adj_info_ph')
            self.v_adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.adj_degree], name='v_adj_info_ph')

            #homogeneous graph information 
            self.u_u_graph_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.num_neigh], name='u_u_graph_ph')
            self.v_v_graph_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.num_neigh], name='v_v_graph_ph')

            self.user_agg_funcs, self.item_agg_funcs = [], []
            if expansion:
                self.u_u_agg_func = DEGC_MyMeanAggregator('u_u_agg_func', self.num_self_neigh, self.input_dim,
                                                    self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                    dropout=self.neigh_drop_rate)
                self.v_v_agg_func = DEGC_MyMeanAggregator('v_v_agg_func', self.num_self_neigh, self.input_dim,
                                                    self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                    dropout=self.neigh_drop_rate)    

                self.u_u_agg_func.extend_homogeneous_conv(self.ex_k)
                self.v_v_agg_func.extend_homogeneous_conv(self.ex_k)

                self.graph_conv_params['u_u_agg_func' + 'output_weight'] = self.u_u_agg_func.output_weight
                self.graph_conv_params['v_v_agg_func' + 'output_weight'] = self.v_v_agg_func.output_weight

                if self.aggregator == 'my_mean':
                    for i in range(1, self.num_layers):
                        self.user_agg_funcs.append(DEGC_MyMeanAggregator('user_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                    self.input_dim, self.layer_dims[i - 1],
                                                                    self.layer_dims[i],
                                                                    activation=self.gcn_act, dropout=self.neigh_drop_rate))
                        self.item_agg_funcs.append(DEGC_MyMeanAggregator('item_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                    self.input_dim, self.layer_dims[i - 1],
                                                                    self.layer_dims[i],
                                                                    activation=self.gcn_act, dropout=self.neigh_drop_rate))  

                        if i == 1:
                            self.user_agg_funcs[0].extend_bottom_conv(self.ex_k)
                            self.item_agg_funcs[0].extend_bottom_conv(self.ex_k)
                        elif i == self.num_layers - 1:
                            self.user_agg_funcs[self.num_layers - 2].extend_top_conv(self.ex_k)
                            self.item_agg_funcs[self.num_layers - 2].extend_top_conv(self.ex_k)
                        else:
                            self.user_agg_funcs[i-1].extend_conv(self.ex_k)
                            self.item_agg_funcs[i-1].extend_conv(self.ex_k)      

                        self.graph_conv_params['user_graph_agg_%d' % i + 'output_weight'] = self.user_agg_funcs[i-1].output_weight
                        self.graph_conv_params['item_graph_agg_%d' % i + 'output_weight'] = self.item_agg_funcs[i-1].output_weight

            elif prediction:
                final = True
                self.u_u_agg_func = DEGC_MyMeanAggregator('u_u_agg_func', self.num_self_neigh, self.input_dim,
                                                    self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                    dropout=self.neigh_drop_rate)
                self.v_v_agg_func = DEGC_MyMeanAggregator('v_v_agg_func', self.num_self_neigh, self.input_dim,
                                                    self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                    dropout=self.neigh_drop_rate)    

                self.u_u_agg_func.load_predict_conv()
                self.v_v_agg_func.load_predict_conv()

                self.graph_conv_params['u_u_agg_func' + 'output_weight'] = self.u_u_agg_func.output_weight
                self.graph_conv_params['v_v_agg_func' + 'output_weight'] = self.v_v_agg_func.output_weight

                if self.aggregator == 'my_mean':
                    for i in range(1, self.num_layers):
                        self.user_agg_funcs.append(DEGC_MyMeanAggregator('user_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                    self.input_dim, self.layer_dims[i - 1],
                                                                    self.layer_dims[i],
                                                                    activation=self.gcn_act, dropout=self.neigh_drop_rate))
                        self.item_agg_funcs.append(DEGC_MyMeanAggregator('item_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                    self.input_dim, self.layer_dims[i - 1],
                                                                    self.layer_dims[i],
                                                                    activation=self.gcn_act, dropout=self.neigh_drop_rate))  

                        self.user_agg_funcs[i-1].load_predict_conv()
                        self.item_agg_funcs[i-1].load_predict_conv()  

                        self.graph_conv_params['user_graph_agg_%d' % i + 'output_weight'] = self.user_agg_funcs[i-1].output_weight
                        self.graph_conv_params['item_graph_agg_%d' % i + 'output_weight'] = self.item_agg_funcs[i-1].output_weight
            else:
                self.u_u_agg_func = DEGC_MyMeanAggregator('u_u_agg_func', self.num_self_neigh, self.input_dim,
                                                    self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                    dropout=self.neigh_drop_rate)
                self.v_v_agg_func = DEGC_MyMeanAggregator('v_v_agg_func', self.num_self_neigh, self.input_dim,
                                                    self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                    dropout=self.neigh_drop_rate)  

                self.u_u_agg_func.load_conv_from_loadcheckpoint(self.load_checkpoint)
                self.v_v_agg_func.load_conv_from_loadcheckpoint(self.load_checkpoint)

                self.graph_conv_params['u_u_agg_func' + 'output_weight'] = self.u_u_agg_func.output_weight
                self.graph_conv_params['v_v_agg_func' + 'output_weight'] = self.v_v_agg_func.output_weight

                if self.aggregator == 'my_mean':
                    for i in range(1, self.num_layers):
                        self.user_agg_funcs.append(DEGC_MyMeanAggregator('user_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                    self.input_dim, self.layer_dims[i - 1],
                                                                    self.layer_dims[i],
                                                                    activation=self.gcn_act, dropout=self.neigh_drop_rate))
                        self.item_agg_funcs.append(DEGC_MyMeanAggregator('item_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                    self.input_dim, self.layer_dims[i - 1],
                                                                    self.layer_dims[i],
                                                                    activation=self.gcn_act, dropout=self.neigh_drop_rate))            
                        
                        if i == self.num_layers - 1:
                            self.user_agg_funcs[i-1].init_conv()
                            self.item_agg_funcs[i-1].init_conv()                      
                        else:
                            self.user_agg_funcs[i-1].load_conv_from_loadcheckpoint(self.load_checkpoint)
                            self.item_agg_funcs[i-1].load_conv_from_loadcheckpoint(self.load_checkpoint)

                        self.graph_conv_params['user_graph_agg_%d' % i + 'output_weight'] = self.user_agg_funcs[i-1].output_weight
                        self.graph_conv_params['item_graph_agg_%d' % i + 'output_weight'] = self.item_agg_funcs[i-1].output_weight

            self.user_agg_funcs = self.user_agg_funcs[::-1]
            self.item_agg_funcs = self.item_agg_funcs[::-1]

            self.bpr_loss, self.reg_loss, self.dist_loss = self.model_fn('model', final=final)
            self.loss = self.bpr_loss + self.reg_loss + self.dist_loss

    def first_build_model(self):
        with self.graph.as_default():
            self.u_id = tf.placeholder(tf.int32, shape=[None, ], name='u_id')
            self.random_ids = tf.placeholder(tf.int32, shape=[None, ], name='random_id')
            self.pos_item_id = tf.placeholder(tf.int32, shape=[None, ], name='pos_item_id')
            self.neg_item_id = tf.placeholder(tf.int32, shape=[None, self.neg_item_num], name='neg_item_id')

            self.u_adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.adj_degree], name='u_adj_info_ph')
            self.v_adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.adj_degree], name='v_adj_info_ph')

            #homogeneous graph information 
            self.u_u_graph_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.num_neigh], name='u_u_graph_ph')
            self.v_v_graph_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.num_neigh], name='v_v_graph_ph')

            self.user_agg_funcs, self.item_agg_funcs = [], []
            self.u_u_agg_func = MyMeanAggregator('u_u_agg_func', self.num_self_neigh, self.input_dim,
                                                self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                dropout=self.neigh_drop_rate)
            self.v_v_agg_func = MyMeanAggregator('v_v_agg_func', self.num_self_neigh, self.input_dim,
                                                self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                dropout=self.neigh_drop_rate)     
            #self.u_u_agg_func.init_conv()
            #self.v_v_agg_func.init_conv()

            self.graph_conv_params['u_u_agg_func' + 'output_weight'] = self.u_u_agg_func.output_weight
            self.graph_conv_params['v_v_agg_func' + 'output_weight'] = self.v_v_agg_func.output_weight      
        
            if self.aggregator == 'my_mean':
                for i in range(1, self.num_layers):
                    self.user_agg_funcs.append(MyMeanAggregator('user_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                self.input_dim, self.layer_dims[i - 1],
                                                                self.layer_dims[i],
                                                                activation=self.gcn_act, dropout=self.neigh_drop_rate))
                    self.item_agg_funcs.append(MyMeanAggregator('item_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                self.input_dim, self.layer_dims[i - 1],
                                                                self.layer_dims[i],
                                                                activation=self.gcn_act, dropout=self.neigh_drop_rate))  

                    #self.user_agg_funcs[i-1].init_conv()
                    #self.item_agg_funcs[i-1].init_conv() 

                    self.graph_conv_params['user_graph_agg_%d' % i + 'output_weight'] = self.user_agg_funcs[i-1].output_weight
                    self.graph_conv_params['item_graph_agg_%d' % i + 'output_weight'] = self.item_agg_funcs[i-1].output_weight          

            self.user_agg_funcs = self.user_agg_funcs[::-1]
            self.item_agg_funcs = self.item_agg_funcs[::-1]

            self.bpr_loss, self.reg_loss, self.dist_loss = self.model_fn('model', final=True)
            self.loss = self.bpr_loss + self.reg_loss + self.dist_loss


    def optimization(self, selective=False):
        with self.graph.as_default():
            if selective:
                all_graph_conv_var = [ var for var in tf.trainable_variables() if '%d'%(self.num_layers-1) in var.name]
            else:
                all_graph_conv_var = [ var for var in tf.trainable_variables() if 'embedding' not in var.name]
            
            l2_losses = []
            for graph_conv_var in all_graph_conv_var:
                l2_losses.append(tf.nn.l2_loss(graph_conv_var))

            self.loss = self.loss + self.l2_lambda * tf.reduce_sum(l2_losses)   

            assert self.ptmzr_argv[0].lower() == 'adam'
            _learning_rate, _epsilon = self.ptmzr_argv[1:3]
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            trainnable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES) 
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon)
                grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list = trainnable_variables)
                self.ptmzr = self.optimizer.apply_gradients(grads_and_vars)
            
            ############ L1 regularization ##########
            l1_var = [ var for var in tf.trainable_variables() if 'embedding' not in var.name]
            l1_op_list = []
            with tf.control_dependencies([self.ptmzr]):
                for var in l1_var:
                    th_t = tf.cast(tf.fill(tf.shape(var), tf.convert_to_tensor(self.l1_lambda)), tf.float64)
                    zero_t = tf.cast(tf.zeros(tf.shape(var)), tf.float64)
                    var_temp = var - (th_t * tf.sign(var))
                    l1_op = var.assign(tf.where(tf.less(tf.abs(var), th_t), zero_t, var_temp))
                    l1_op_list.append(l1_op)
           
           ############# Group sparse regularization ###########
            GL_var = [var for var in tf.trainable_variables() if 'new' in var.name and 'top' not in var.name and 'embedding' not in var.name]
            gl_op_list = []
            with tf.control_dependencies([self.ptmzr]):
                for var in GL_var:
                    g_sum = tf.sqrt(tf.reduce_sum(tf.square(var), 0))
                    th_t = tf.cast(self.gl_lambda, tf.float64)
                    gw = []
                    for i in range(var.get_shape()[1]):
                        temp_gw = var[:, i] - (th_t * var[:, i] / g_sum[i])
                        gw_gl = tf.where(tf.less(g_sum[i], th_t), tf.cast(tf.zeros(tf.shape(var[:, i])), tf.float64), temp_gw)
                        gw.append(gw_gl)
                    gl_op = var.assign(tf.stack(gw, 1))
                    gl_op_list.append(gl_op)

            with tf.control_dependencies(l1_op_list + gl_op_list):
                self.opt = tf.no_op()                    


    def first_optimization(self):
        with self.graph.as_default():
            assert self.ptmzr_argv[0].lower() == 'adam'
            _learning_rate, _epsilon = self.ptmzr_argv[1:3]
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            trainnable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES) 
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon)
                grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list = trainnable_variables)
                self.ptmzr = self.optimizer.apply_gradients(grads_and_vars)
 

    def get_params(self, sess):
        """ Access the Graph Convolution Parameters"""
        params_dict = dict()
        for param_name, param in self.graph_conv_params.items():
            param_value = sess.run(param)
            params_dict[param_name] = param_value

        return params_dict


    def load_params(self, params):
        self.graph_conv_params = dict()
        for name, param in params.items():
            variable = tf.get_variable(name, initializer=param)
            self.graph_conv_params[name] = variable


    def selective_learning(self, selected_params):
        tf.reset_default_graph()
        with self.graph.as_default():
            self.u_id = tf.placeholder(tf.int32, shape=[None, ], name='u_id')
            self.random_ids = tf.placeholder(tf.int32, shape=[None, ], name='random_id')
            self.pos_item_id = tf.placeholder(tf.int32, shape=[None, ], name='pos_item_id')
            self.neg_item_id = tf.placeholder(tf.int32, shape=[None, self.neg_item_num], name='neg_item_id')

            self.u_adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.adj_degree], name='u_adj_info_ph')
            self.v_adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.adj_degree], name='v_adj_info_ph')

            #homogeneous graph information 
            self.u_u_graph_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.num_neigh], name='u_u_graph_ph')
            self.v_v_graph_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, self.num_neigh], name='v_v_graph_ph')

            self.user_agg_funcs, self.item_agg_funcs = [], []
            self.u_u_agg_func = DEGC_MyMeanAggregator('u_u_agg_func', self.num_self_neigh, self.input_dim,
                                                self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                dropout=self.neigh_drop_rate)
            self.v_v_agg_func = DEGC_MyMeanAggregator('v_v_agg_func', self.num_self_neigh, self.input_dim,
                                                self.input_dim, self.layer_dims[-1], activation=self.gcn_act,
                                                dropout=self.neigh_drop_rate)   

            self.u_u_agg_func.load_conv_from_selected(selected_params['u_u_agg_func' + 'output_weight'])   
            self.v_v_agg_func.load_conv_from_selected(selected_params['v_v_agg_func' + 'output_weight']) 

            self.graph_conv_params['u_u_agg_func' + 'output_weight'] = self.u_u_agg_func.output_weight
            self.graph_conv_params['v_v_agg_func' + 'output_weight'] = self.v_v_agg_func.output_weight
            
            if self.aggregator == 'my_mean':
                for i in range(1, self.num_layers):
                    self.user_agg_funcs.append(DEGC_MyMeanAggregator('user_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                self.input_dim, self.layer_dims[i - 1],
                                                                self.layer_dims[i],
                                                                activation=self.gcn_act, dropout=self.neigh_drop_rate))
                    self.item_agg_funcs.append(DEGC_MyMeanAggregator('item_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                self.input_dim, self.layer_dims[i - 1],
                                                                self.layer_dims[i],
                                                                activation=self.gcn_act, dropout=self.neigh_drop_rate))  
                     
                    self.user_agg_funcs[i-1].load_conv_from_selected(selected_params['user_graph_agg_%d' % i + 'output_weight'])   
                    self.item_agg_funcs[i-1].load_conv_from_selected(selected_params['item_graph_agg_%d' % i + 'output_weight'])    

                    self.graph_conv_params['user_graph_agg_%d' % i + 'output_weight'] = self.user_agg_funcs[i-1].output_weight
                    self.graph_conv_params['item_graph_agg_%d' % i + 'output_weight'] = self.item_agg_funcs[i-1].output_weight                                                

            self.user_agg_funcs = self.user_agg_funcs[::-1]
            self.item_agg_funcs = self.item_agg_funcs[::-1]

            self.bpr_loss, self.reg_loss, self.dist_loss = self.model_fn('model', final=False)
            self.loss = self.bpr_loss + self.reg_loss + self.dist_loss


    






            
