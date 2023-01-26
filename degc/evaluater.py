# -*- coding: utf-8 -*-
"""
Implementation of evaluate_MGCCF and evaluate_NGCF with tensorflow

@author: Bowei He
"""

"""
Load the model parameters generated from the base model training stream (run_RCGF.py), evaluate the performance of each action
via 'pruning+expanding parameters + retraining + validating the trained model on the validation data'
"""
import tensorflow as tf
import numpy as np
import pdb
from run_baselines_segments import *
from degc.dynamic_MGCCF import Dynamic_MGCCF
from degc.dynamic_NDCG import Dynamic_NDCG
from degc.dynamic_LightGCN import Dynamic_LightGCN



class evaluater_MLP:
    def __init__(self,task_list,args):
        self.epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.task_list = task_list
        self.stamps={}
        
    def evaluate_action(self,var_list,actions,task_id):
        with tf.Graph().as_default() as g:
            self.sess = tf.compat.v1.Session(graph=g)
            self.stamps[task_id-1] = [_.shape for _ in var_list]
            self.task_id = task_id
            with tf.name_scope("model"):
                self.x = tf.placeholder(tf.float32,shape=[None,784]) 
                self.y = tf.placeholder(tf.float32,shape=[None,10])
                fc1 = tf.Variable(tf.concat([var_list[0],tf.truncated_normal((var_list[0].shape[0],actions[0]),stddev=0.01)],axis=1))
                b1 = tf.Variable(tf.concat([var_list[1],tf.constant(0.1,shape=(actions[0],))],axis=0))
                mask_fc1 = np.concatenate([np.zeros_like(var_list[0]),np.ones((var_list[0].shape[0],actions[0]))],axis=1)
                mask_b1 = np.concatenate([np.zeros_like(var_list[1]),np.ones((actions[0]))],axis=0)

                
                old_shape = var_list[2].shape
                value = tf.concat([var_list[2],tf.truncated_normal((actions[0],old_shape[1]),stddev=0.01)],axis=0)
                fc2 = tf.Variable(tf.concat([value,tf.truncated_normal((actions[0]+old_shape[0],actions[1]),stddev=0.01)],axis=1))
                b2 = tf.Variable(tf.concat([var_list[3],tf.constant(0.1,shape=(actions[1],))],axis=0))
                mask_fc2 = np.concatenate([np.concatenate([np.zeros_like(var_list[2]),np.ones((actions[0],old_shape[1]))],axis=0),np.ones((actions[0]+old_shape[0],actions[1]))],axis=1)
                mask_b2 = np.concatenate([np.zeros_like(var_list[3]),np.ones((actions[1],))],axis=0)

                fc3 = tf.Variable(tf.truncated_normal((var_list[4].shape[0]+actions[1],var_list[4].shape[1])))
                mask_fc3 = np.ones_like(fc3)
                b3 = tf.Variable(tf.constant(0.1,shape=(var_list[4].shape[1],)))
                mask_b3 = np.ones_like(b3)
            total_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(self.x,fc1,b1,name="fc1"))
            h_fc2 = tf.nn.relu(tf.nn.xw_plus_b(h_fc1,fc2,b2,name="fc2"))
            h_fc3 = tf.nn.xw_plus_b(h_fc2,fc3,b3,name="fc3")
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y,logits = h_fc3)) + 0.0001*(tf.nn.l2_loss(fc1) + tf.nn.l2_loss(fc2) + tf.nn.l2_loss(fc3))
            
            if self.optimizer=="adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif self.optimizer=="rmsprop":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.optimizer=="sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                raise Exception("please choose one optimizer")
            total_mask = [mask_fc1,mask_b1,mask_fc2,mask_b2,mask_fc3,mask_b3]
            grads_and_vars = optimizer.compute_gradients(loss,var_list= total_theta)
            grads_and_vars2 = self.apply_prune_on_grads(grads_and_vars, total_mask)
            train_step = optimizer.apply_gradients(grads_and_vars2)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y,axis=1),tf.argmax(h_fc3,axis=1)),tf.float32))
            self.sess.run(tf.global_variables_initializer())
            
            l = len(self.task_list[0][1])
            for epoch in range(self.epochs):
                flag=0
                for _ in range(l//self.batch_size+1):
                    batch_xs, batch_ys = (self.task_list[task_id][0][flag:flag+self.batch_size],
                                          self.task_list[task_id][1][flag:flag+self.batch_size])
                    flag+=self.batch_size
                    self.sess.run(train_step,feed_dict={self.x:batch_xs,self.y:batch_ys})
                accuracy_val = self.sess.run(accuracy, feed_dict={self.x:self.task_list[task_id][2],
                                                                  self.y:self.task_list[task_id][3]})
                accuracy_test = self.sess.run(accuracy,feed_dict={self.x:self.task_list[task_id][4],
                                                                  self.y:self.task_list[task_id][5]})
                if epoch%4==0 or epoch==self.epochs-1:
                    print("task:%s,test accuracy:%s"%(task_id,accuracy_test))
            self.var_list = self.sess.run(total_theta)
            self.stamps[task_id]=[_.shape for _ in self.var_list]
            self.sess.close()
            return (accuracy_val,accuracy_test)

    def conv2d(self,x, W): 
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    def apply_prune_on_grads(self,grads_and_vars, total_mask):
        for i in range(0,len(total_mask),2):
            grads_and_vars[i] = (tf.multiply(grads_and_vars[i][0], total_mask[i]),grads_and_vars[i][1])
            grads_and_vars[i+1] = (tf.multiply(grads_and_vars[i+1][0], total_mask[i+1]),grads_and_vars[i+1][1])
        return grads_and_vars

class evaluater:
    """
    The evaluation class which wraps the train_model and evaluate_model functions to evaluate
    the actions provided by the RL-based GNN structure controller.

    Base_model: MGCCF, NGCF, LightGCN
    """
    def __init__(self, args, segment, train_info, val_info, test_info, old_train_set, old_train_matrix,\
                 n_epoch, n_old_user=0, n_old_item=0, node_deg_delta=None, logger=None, load_checkpoint='', save_checkpoint='', graph_path=None):
        self.args = args
        self.base_model = args.base_model
        self.segment = segment
        self.train_set, self.n_user_train, self.n_item_train, self.train_matrix = train_info
        self.val_info = val_info
        self.val_set, self.n_user_val, self.n_item_val, self.val_matrix = val_info
        self.test_info = test_info
        self.test_set, self.n_user_test, self.n_item_test, self.test_matrix = test_info
        self.old_train_set = old_train_set
        self.old_train_matrix = old_train_matrix
        self.n_epoch = n_epoch
        self.n_old_user = n_old_user
        self.n_old_item = n_old_item
        self.node_deg_delta = node_deg_delta
        self.logger = logger
        self.load_checkpoint = load_checkpoint
        self.save_ckpt = save_checkpoint
        self.save_checkpoint = save_checkpoint
        self.graph_path = graph_path
        self.time_info = []

        if self.base_model == 'MGCCF':
            self.user_self_neighs, _ = load_self_neighbours(self.graph_path[0], 'train', self.n_user_train, args.num_neigh, self.train_matrix)
            self.item_self_neighs, _ = load_self_neighbours(self.graph_path[1], 'train', self.n_item_train, args.num_neigh, self.train_matrix.transpose())
            self.user_self_neighs_val, _ = load_self_neighbours(self.graph_path[0], 'val', self.n_user_val, args.num_neigh, self.val_matrix)
            self.item_self_neighs_val, _ = load_self_neighbours(self.graph_path[1], 'val', self.n_item_val, args.num_neigh, self.val_matrix.transpose())
            self.user_self_neighs_test, _ = load_self_neighbours(self.graph_path[0], 'test', self.n_user_test, args.num_neigh, self.test_matrix)
            self.item_self_neighs_test, _ = load_self_neighbours(self.graph_path[1], 'test', self.n_item_test, args.num_neigh, self.test_matrix.transpose())

        #prepare train data
        self.u_adj_dict_train, self.i_adj_dict_train = sparse_adj_matrix_to_dicts(self.train_matrix)
        self.u_adj_list_train, self.i_adj_list_train = pad_adj(self.u_adj_dict_train, args.max_degree, self.n_item_train), pad_adj(self.i_adj_dict_train, args.max_degree, self.n_user_train)  

        self.u_adj_dict_val, self.i_adj_dict_val = sparse_adj_matrix_to_dicts(self.val_matrix)
        self.u_adj_list_val, self.i_adj_list_val = pad_adj(self.u_adj_dict_val, args.max_degree, self.n_item_val), pad_adj(self.i_adj_dict_val, args.max_degree, self.n_user_val)

        self.u_adj_dict_test, self.i_adj_dict_test = sparse_adj_matrix_to_dicts(self.test_matrix)
        self.u_adj_list_test, self.i_adj_list_test = pad_adj(self.u_adj_dict_test, args.max_degree, self.n_item_test), pad_adj(self.i_adj_dict_test, args.max_degree, self.n_user_test)     
        
        """
        self.epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.stamps={}
        """

    def evaluate_actions(self, actions=None, embedded_dimension=[128, 128, 128]): 

        self.logger.write("actions: " + str(actions) + '\n')
        print('actions', actions)
        self.embedded_dimension = embedded_dimension


        if self.base_model == 'MGCCF':
            model = Dynamic_MGCCF(self.args.new_node_init,
                    self.train_matrix,
                    [self.n_user_train, self.n_old_user],
                    [self.n_item_train, self.n_old_item],
                    self.load_checkpoint,
                    [eval(self.args.embedded_dimension)[0], self.n_user_train, self.n_item_train],
                    self.embedded_dimension[1:],
                    self.args.max_degree,
                    eval(self.args.gcn_sample),
                    ['adam', self.args.learning_rate, self.args.epsilon],
                    'my_mean',
                    self.args.activation,
                    self.args.neighbor_dropout,
                    self.args.l2,
                    self.args.dist_embed,
                    num_self_neigh = self.args.num_self_neigh,
                    neg_item_num = self.args.num_neg,
                    pretrain_data = self.args.pretrain_data if self.segment > 0 else False,
                    num_neigh = self.args.num_neigh,
                    actions = actions
            )

        else:
            NotImplementedError
        """
        elif self.base_model == 'NGCF':
            if self.segment > 0:
                model = Dynamic_NGCF(

                    actions = actions
                )
            elif:
                model = Dynamic_NGCF(

                    actions = None
                ) 
        elif self.base_model == 'LightGCN':
            if self.segment > 0:
                model = Dynamic_LightGCN(
                    actions = actions
                )
            elif:
                model = Dynamic_LightGCN(
                    actions = None
                )

        else:
            NotImplementedError
        """
        num_pairs = 0
        for i in range(len(self.train_set)):
            num_pairs += len(self.train_set[i])
        num_iter = int(num_pairs / self.args.batch_pairs) + 1
        iter_time = []

        sampler = WarpSampler(self.train_set,
                              self.n_item_train,
                              batch_size = self.args.batch_pairs,
                              n_negative = self.args.num_neg,
                              n_workers = 2,
                              check_negative = True      
        )

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        with model.graph.as_default():
            saver = tf.train.Saver(max_to_keep=100)

        with tf.compat.v1.Session(graph=model.graph, config=sess_config) as sess:
            #initialize variables
            sess.run(tf.global_variables_initializer())
            
            _epoch = 0
            best_valid_recall20, best_valid_ndcg20, best_valid_epoch, best_test_recall20, best_test_ndcg20 = 0., 0., 0., 0., 0.
            early_stop_flag = 0
            mse_user_reg, mse_item_reg = None, None
            
            time_info_training = 0
            time_info_eval = 0
            time_info_sampling = 0

            while _epoch <= self.n_epoch:
                self.time_info.append(('start epoch' + str(_epoch) + ' training', time.time()))

                if _epoch % 1 == 0:
                    time_info_eval_start = time.time()
                    if self.base_model == 'MGCCF':
                        precision, v_recall, MAP, v_ndcg, _, _ = evaluate_model(sess, self.base_model, model,  self.val_info, self.train_matrix, self.u_adj_list_val, self.i_adj_list_val,
                                                                            self.user_self_neighs_val, self.item_self_neighs_val, n_batch_users=self.args.batch_evaluate)
                    elif self.base_model == 'NGCF':
                        precision, v_recall, MAP, v_ndcg, _, _ = evaluate_model(sess, self.base_model, model,  self.val_info, self.train_matrix, self.u_adj_list_val, self.i_adj_list_val,
                                                                            n_batch_users=self.args.batch_evaluate)                   
                    elif self.base_model == 'LightGCN':
                        precision, v_recall, MAP, v_ndcg, _, _ = evaluate_model(sess, self.base_model, model, self.val_info, self.train_matrix, self.u_adj_list_val, self.i_adj_list_val,
                                                                            n_batch_users=self.args.batch_evaluate)

                    else:
                        NotImplementedError

                    write_prediction_to_logger(self.logger, precision, v_recall, MAP, v_ndcg, _epoch, 'validation set')

                    if v_recall[-1] >= best_valid_recall20:
                        #accelerate: only check testset when finding best model on validation dataset
                        if self.base_model == 'MGCCF':
                            precision, t_recall, MAP, t_ndcg, _, _ = evaluate_model(sess, self.base_model, model, self.test_info, self.train_matrix,
                                                                                self.u_adj_list_test, self.i_adj_list_test, self.user_self_neighs_test,
                                                                                self.item_self_neighs_test, n_batch_users=self.args.batch_evaluate)
                        elif self.base_model == 'NGCF':
                            precision, t_recall, MAP, t_ndcg, _, _ = evaluate_model(sess, self.base_model, model, self.test_info, self.train_matrix,
                                                                                self.u_adj_list_test, self.i_adj_list_test, n_batch_users=self.args.batch_evaluate)  
                        elif self.base_model == 'LightGCN':
                            precision, t_recall, MAP, t_ndcg, _, _ = evaluate_model(sess, self.base_model, model, self.test_info, self.train_matrix,
                                                                                self.u_adj_list_test, self.i_adj_list_test, n_batch_users=self.args.batch_evaluate)
                        else:
                            NotImplementedError                      
                        write_prediction_to_logger(self.logger, precision, t_recall, MAP, t_ndcg, _epoch, 'test set')

                        early_stop_flag = 0
                        best_valid_recall20 = v_recall[-1]
                        best_valid_ndcg20 = v_ndcg[-1]
                        best_valid_epoch = _epoch
                        best_test_recall20 = t_recall[-1]
                        best_test_ndcg20 = t_ndcg[-1]  

                        if self.save_checkpoint != "":
                            # save embedding for next time segement
                            saver.save(sess, self.save_checkpoint)   

                    early_stop_flag += 1         
                    if early_stop_flag > self.args.patience and _epoch > self.args.min_epoch:
                        if self.logger is not None:
                            self.logger.write('early stop triggered at epoch: ' + str(_epoch) + '\n')
                        else:
                            print('early stop triggered at epoch' + str(_epoch))
                        break

                    time_info_eval_end = time.time()
                    time_info_eval += time_info_eval_end - time_info_eval_start

                _epoch += 1

                if _epoch > self.n_epoch:
                    break

                for iter in range(0, num_iter):
                    time_info_sampling_start = time.time()
                    user_pos, neg_samples = sampler.next_batch()

                    iter_start = time.time()
                    time_info_sampling += iter_start - time_info_sampling_start
                    if self.base_model == 'MGCCF':
                        feed_dict = {model.u_id: user_pos[:, 0],
                                    model.pos_item_id: user_pos[:, 1],
                                    model.neg_item_id: neg_samples,
                                    model.u_adj_info_ph: self.u_adj_list_train,
                                    model.v_adj_info_ph: self.i_adj_list_train,
                                    model.u_u_graph_ph: self.user_self_neighs,
                                    model.v_v_graph_ph: self.item_self_neighs}
                    elif self.base_model == 'NGCF':
                        feed_dict = {model.u_id: user_pos[:, 0],
                                    model.pos_item_id: user_pos[:, 1],
                                    model.neg_item_id: neg_samples,
                                    model.u_adj_info_ph: self.u_adj_list_train,
                                    model.v_adj_info_ph: self.i_adj_list_train}
                    elif self.base_model == 'LightGCN':
                        feed_dict = {model.u_id: user_pos[:, 0],
                                    model.pos_item_id: user_pos[:, 1],
                                    model.neg_item_id: neg_samples,
                                    model.u_adj_info_ph: self.u_adj_list_train,
                                    model.v_adj_info_ph: self.i_adj_list_train}
                    else:
                        NotImplementedError   

                    _, bpr_loss, l2_reg, dist_loss  = sess.run([model.ptmzr,
                                                                model.bpr_loss,
                                                                model.reg_loss,
                                                                model.dist_loss],                                                                                                                                                                                                             
                                                                feed_dict=feed_dict)                 

                    print('Epoch ', '%04d' % _epoch, 'iter ', '%02d' % iter,
                        'bpr_loss=', '{:.5f}, reg_loss= {:.5f}, dist_loss= {:.5f},'.format(bpr_loss, l2_reg, dist_loss),
                        'cost {:.4f} seconds'.format(time.time() - iter_start))
                    iter_time.append(time.time() - iter_start)
                    self.logger.write(f"bpr: {np.round(bpr_loss, 5)}, reg: {np.round(l2_reg, 5)}, dist: {np.round(dist_loss, 5)}" + '\n')
                
                self.time_info.append(('finish epoch ' + str(_epoch) + 'training', time.time()))
                time_info_training = sum(iter_time)       
            self.time_info.append(('finish final epoch training', time.time()))
            self.time_info.append(('total training time', time_info_training))
            self.time_info.append(('total eval time', time_info_eval))
            self.time_info.append(('total sampling time', time_info_sampling))

        sampler.close()
        if self.args.log_name:
            self.logger.write("training time: " + str(sum(iter_time)) + '\n')
            self.logger.write('best_valid_epoch, best_valid_recall20, best_valid_ndcg20, best_test_recall20, best_test_ndcg20' + '\n')
            self.logger.write(str([best_valid_epoch, best_valid_recall20, best_valid_ndcg20, best_test_recall20, best_test_ndcg20]) + '\n')
        else:
            print("training time: " + str(sum(iter_time)) + '\n')
            print('best_valid_epoch, best_valid_recall20, best_valid_ndcg20, best_test_recall20, best_test_ndcg20')
            print(str([best_valid_epoch, best_valid_recall20, best_valid_ndcg20, best_test_recall20, best_test_ndcg20]))      

        return best_valid_recall20, best_test_recall20, best_valid_ndcg20, best_test_ndcg20    

