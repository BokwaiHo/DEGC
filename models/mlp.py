from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import os

class MLP(object):

    def __init__(self, u_emb_size, i_emb_size, n_pos, n_neg_per_user):

        self.user_input = tf.placeholder(tf.float64, shape=[None, u_emb_size*2])
        self.item_input = tf.placeholder(tf.float64, shape=[None, i_emb_size*2])

        self.w_user = tf.get_variable('w_user', dtype=tf.float64, shape=[u_emb_size*2, u_emb_size])
        self.w_item = tf.get_variable('w_item', dtype=tf.float64, shape=[i_emb_size*2, i_emb_size])

        self.b_user = tf.get_variable('b_user', dtype=tf.float64, shape=[u_emb_size])
        self.b_item = tf.get_variable('b_item', dtype=tf.float64, shape=[i_emb_size])

        u_rep = tf.add(tf.matmul(self.user_input, self.w_user), self.b_user)
        i_rep = tf.add(tf.matmul(self.item_input, self.w_item), self.b_item)
        
        n_neg = n_pos * n_neg_per_user

        pos_i_rep = i_rep[:n_pos]
        pos_rating = tf.reduce_sum(tf.multiply(u_rep, pos_i_rep), 1) # n_pos
        pos_rating = tf.expand_dims(pos_rating, 1) # n_pos, 1
        pos_rating = tf.tile(pos_rating, [1, n_neg_per_user]) # n_pos, n_neg
        pos_rating = tf.reshape(pos_rating, [n_neg, 1]) # n_pos*n_neg, 1
    
        neg_i_rep = tf.reshape(i_rep[n_pos:], [n_pos, n_neg_per_user, -1])
        expand_u_rep = tf.expand_dims(u_rep, 1) 
        expand_u_rep = tf.tile(expand_u_rep, [1, n_neg_per_user, 1])
        neg_rating = tf.reduce_sum(tf.multiply(expand_u_rep, neg_i_rep), 2)
        neg_rating = tf.reshape(neg_rating, [n_neg, 1])

        bpr_loss = pos_rating - neg_rating
        bpr_loss = tf.nn.sigmoid(bpr_loss)
        bpr_loss = -tf.log(bpr_loss)
        bpr_loss = tf.reduce_sum(bpr_loss)

        bpr_loss += 0.02 * (tf.nn.l2_loss(u_rep) + tf.nn.l2_loss(i_rep))

        self.u_rep, self.i_rep, self.bpr_loss = u_rep, i_rep, bpr_loss
        self.optimizer = tf.train.AdamOptimizer().minimize(self.bpr_loss)

    def predit(self, n_new_user, n_new_item):
        u_rep = tf.add(tf.matmul(self.user_input, self.w_user), self.b_user)
        i_rep = tf.add(tf.matmul(self.item_input, self.w_item), self.b_item)

        if n_new_user > 0:
            try:
                new_user_rep = tf.get_default_graph().get_tensor_by_name("new_user_rep:0")
            except KeyError:
                print("!!!!!!only once!!!!!")
                new_user_rep = tf.random.uniform([n_new_user, u_rep.shape[1].value], dtype=tf.float64, name='new_user_rep')
            u_rep = tf.concat([u_rep, new_user_rep], axis=0)

        if n_new_item > 0:
            try:
                new_item_rep = tf.get_default_graph().get_tensor_by_name("new_item_rep:0")
            except KeyError:
                print("!!!!!!only once!!!!!")
                new_item_rep = tf.random.uniform([n_new_item, i_rep.shape[1].value], dtype=tf.float64, name='new_item_rep')
            i_rep = tf.concat([i_rep, new_item_rep], axis=0)

        rating_score = tf.matmul(u_rep, i_rep, transpose_b=True)

        return rating_score