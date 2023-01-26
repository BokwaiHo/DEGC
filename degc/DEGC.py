import tensorflow as tf
import numpy as np
from collections import defaultdict
import pdb
from run_baselines_segments import *
# -*- coding: utf-8 -*-
"""
Implementation of DEGC algorithm with tensorflow

@author: Bowei He
"""

"""
Load the model parameters generated from the base model training stream (run_DEGC.py), execute the DEGC algorithm via
 'pruning+ expanding parameters + retraining'
"""
import tensorflow as tf
import numpy as np
import pdb
from run_baselines_segments import *
from degc.DEGC_MGCCF import DEGC_MGCCF
from degc.dynamic_NDCG import Dynamic_NDCG
from degc.dynamic_LightGCN import Dynamic_LightGCN


class DEGC:
    """
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
        self.graph_conv_params = dict()

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
        
        # GNN networks
        self.cur_params, self.prev_params = dict(), dict()
        """
        self.epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.stamps={}
        """

    def train_model(self, model, selective=False, final=False): 
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

                    if v_recall[-1] >= best_valid_recall20 and final == True:
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

                        if self.save_checkpoint != "" and final == True:
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

            self._vars = [(var.name, sess.run(var)) for var in tf.trainable_variables() if 'agg' in var.name]
            self.graph_conv_params = model.get_params(sess)

        tf.reset_default_graph()
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
    

    def test_model(self, model):

        sess_config = tf.compat.v1.ConfigProto()

        sess_config.gpu_options.allow_growth = True 

        with tf.compat.v1.Session(graph=model.graph, config=sess_config) as sess:
            #initialize variables
            sess.run(tf.global_variables_initializer())
            best_test_recall20, best_test_ndcg20 = 0., 0.

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


            best_test_recall20 = t_recall[-1]
            best_test_ndcg20 = t_ndcg[-1]
        

        if parser.log_name:
            logger.write('best_test_recall20, best_test_ndcg20' + '\n')
            logger.write(str([best_test_recall20, best_test_ndcg20]) + '\n')
        else:
            print('best_test_recall20, best_test_ndcg20')
            print(str([best_test_recall20, best_test_ndcg20]))

        return best_test_recall20, best_test_ndcg20  

    def DEGC_main(self):

        # define model
        expansion_layer = []
        self.expansion_layer = [0, 0]

        if self.segment == 0:
            if self.base_model == 'MGCCF':
                model = DEGC_MGCCF(self.segment,
                        self.args.new_node_init,
                        self.train_matrix,
                        [self.n_user_train, self.n_old_user],
                        [self.n_item_train, self.n_old_item],
                        self.load_checkpoint,
                        [eval(self.args.embedded_dimension)[0], self.n_user_train, self.n_item_train],
                        eval(self.args.embedded_dimension)[1:],
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
                        num_neigh = self.args.num_neigh
                )

            else:
                NotImplementedError
            
            model.first_build_model()
            model.first_optimization()
            best_valid_recall20, best_test_recall20, best_valid_ndcg20, best_test_ndcg20 = self.train_model(model, final=True)

            return best_valid_recall20, best_test_recall20, best_valid_ndcg20, best_test_ndcg20

        else:
        # execute DEGC
            print('Begin DEGC')
            print('######## Graph Convolution Pruning (prune outdated STP filters) + Selective Training (refine useful LTP filters) ########')
            if self.base_model == 'MGCCF':
                model = DEGC_MGCCF(self.segment,
                        self.args.new_node_init,
                        self.train_matrix,
                        [self.n_user_train, self.n_old_user],
                        [self.n_item_train, self.n_old_item],
                        self.load_checkpoint,
                        [eval(self.args.embedded_dimension)[0], self.n_user_train, self.n_item_train],
                        eval(self.args.embedded_dimension)[1:],
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
                        num_neigh = self.args.num_neigh
                )

            else:
                NotImplementedError
            
            model.build_model()
            model.optimization(selective=True)
            _, _, _, _ = self.train_model(model, selective=True, final=False)

            params = self.graph_conv_params
            self.destroy_graph()
            
            # select the units

            selected_params = dict()
            all_indices = defaultdict(list)  # nonzero unis
            
            ## u-i graph convolution
            for i in range(model.num_layers-1, 0, -1):
                if i == model.num_layers - 1:
                    # user graph conv
                    output_weight_user = params['user_graph_agg_%d' % i + 'output_weight']
                    for j in range(output_weight_user.shape[0]):
                        if np.count_nonzero(output_weight_user[j]) != 0:
                            all_indices['user_graph_agg_%d' % i + 'output_weight'].append(j)
                    all_indices['user_graph_agg_%d' % i + 'output_weight'] = np.array(all_indices['user_graph_agg_%d' % i + 'output_weight'])
                    selected_params['user_graph_agg_%d' % i + 'output_weight'] = output_weight_user[np.ix_(all_indices['user_graph_agg_%d' % i + 'output_weight'])]
                    # item graph conv
                    output_weight_item = params['item_graph_agg_%d' % i + 'output_weight']
                    for j in range(output_weight_item.shape[0]):
                        if np.count_nonzero(output_weight_item[j]) != 0:
                            all_indices['item_graph_agg_%d' % i + 'output_weight'].append(j)
                    all_indices['item_graph_agg_%d' % i + 'output_weight'] = np.array(all_indices['item_graph_agg_%d' % i + 'output_weight'])
                    selected_params['item_graph_agg_%d' % i + 'output_weight'] = output_weight_item[np.ix_(all_indices['item_graph_agg_%d' % i + 'output_weight'])]

                else:
                    # user graph conv
                    output_weight_user = params['user_graph_agg_%d' % i + 'output_weight']
                    top_user_indices = all_indices['user_graph_agg_%d' % (i+1) + 'output_weight'] 
                    top_user_indices = top_user_indices [ top_user_indices >= model.input_dim] - model.input_dim
                    print('top_user_indices', top_user_indices)
                    for j in range(output_weight_user.shape[0]):
                        if np.count_nonzero(output_weight_user[j, top_user_indices]) != 0 or i == 1:
                            all_indices['user_graph_agg_%d' % i + 'output_weight'].append(j)
                    all_indices['user_graph_agg_%d' % i + 'output_weight'] = np.array(all_indices['user_graph_agg_%d' % i + 'output_weight'])
                    selected_params['user_graph_agg_%d' % i + 'output_weight'] = output_weight_user[np.ix_(all_indices['user_graph_agg_%d' % i + 'output_weight'], top_user_indices)]
                    # item graph conv
                    output_weight_item = params['item_graph_agg_%d' % i + 'output_weight']
                    top_item_indices = all_indices['item_graph_agg_%d' % (i+1) + 'output_weight']
                    top_item_indices = top_item_indices [ top_item_indices >= model.input_dim] - model.input_dim
                    for j in range(output_weight_item.shape[0]):
                        if np.count_nonzero(output_weight_item[j, top_item_indices]) != 0 or i == 1:
                            all_indices['item_graph_agg_%d' % i + 'output_weight'].append(j)
                    all_indices['item_graph_agg_%d' % i + 'output_weight'] = np.array(all_indices['item_graph_agg_%d' % i + 'output_weight'])
                    selected_params['item_graph_agg_%d' % i + 'output_weight'] = output_weight_item[np.ix_(all_indices['item_graph_agg_%d' % i + 'output_weight'], top_item_indices)]

            ## u-u and i-i graph convolutions
            selected_params['u_u_agg_func' + 'output_weight'] = params['u_u_agg_func' + 'output_weight']
            selected_params['v_v_agg_func' + 'output_weight'] = params['v_v_agg_func' + 'output_weight']
            
            # learn only selected graph convolution params
            self.destroy_graph()
            model.selective_learning(selected_params)
            model.optimization()

            _, _, _, _ = self.train_model(model, final=False)


            #for item in self.graph_conv_params:
            #    key, values = item
            #    selected_params[key] = values

            for key, value in self.graph_conv_params.items():
                selected_params[key] = value
            
            # union (can consider delete this segment)
            for i in range(model.num_layers-1, 0, -1):
                if i == model.num_layers - 1:
                    # user
                    temp_weight_user = params['user_graph_agg_%d' % i + 'output_weight']
                    temp_weight_user[np.ix_(all_indices['user_graph_agg_%d' % i + 'output_weight'])] = selected_params['user_graph_agg_%d' % i + 'output_weight']
                    params['user_graph_agg_%d' % i + 'output_weight'] = temp_weight_user

                    # item
                    temp_weight_item = params['item_graph_agg_%d' % i + 'output_weight']
                    temp_weight_item[np.ix_(all_indices['item_graph_agg_%d' % i + 'output_weight'])] = selected_params['item_graph_agg_%d' % i + 'output_weight']
                    params['item_graph_agg_%d' % i + 'output_weight'] = temp_weight_item
                else:
                    # user
                    temp_weight_user = params['user_graph_agg_%d' % i + 'output_weight']
                    top_user_indices = all_indices['user_graph_agg_%d' % (i+1) + 'output_weight']
                    top_user_indices = top_user_indices[ top_user_indices >= model.input_dim] - model.input_dim
                    temp_weight_user[np.ix_(all_indices['user_graph_agg_%d' % i + 'output_weight'], top_user_indices)] = selected_params['user_graph_agg_%d' % i + 'output_weight']
                    params['user_graph_agg_%d' % i + 'output_weight'] = temp_weight_user

                    #item 
                    temp_weight_item = params['item_graph_agg_%d' % i + 'output_weight']
                    top_item_indices = all_indices['item_graph_agg_%d' % (i+1) + 'output_weight']
                    top_item_indices = top_item_indices[ top_item_indices >= model.input_dim] - model.input_dim
                    temp_weight_item[np.ix_(all_indices['item_graph_agg_%d' % i + 'output_weight'], top_item_indices)] = selected_params['item_graph_agg_%d' % i + 'output_weight']
                    params['item_graph_agg_%d' % i + 'output_weight'] = temp_weight_item

            params['u_u_agg_func' + 'output_weight'] = selected_params['u_u_agg_func' + 'output_weight']
            params['v_v_agg_func' + 'output_weight'] = selected_params['v_v_agg_func' + 'output_weight']

                    

            print('###### Graph Convolution Expanding (add STP filters for extracting present STP) + Prune(help determine the newly added LTP filter number) ######')    

            self.destroy_graph()

            model.load_params(params)
            model.build_model(expansion=True)

            _, _, _, _ = self.train_model(model, final=False)
            params = self.graph_conv_params

            # prune useless newly added filters
            for i in range(model.num_layers-2, 0, -1):
                # user
                prev_output_user = params['user_graph_agg_%d' % i + 'output_weight']
                useless = []
                for j in range(prev_output_user.shape[1] - model.ex_k, prev_output_user.shape[1]):
                    if np.count_nonzero(prev_output_user[:, j]) == 0:
                        useless.append(j)
                useless = np.array(useless)
                cur_output_user = np.delete(prev_output_user, useless, axis=1)
                params['user_graph_agg_%d' % i + 'output_weight'] = cur_output_user
                prev_output_user = params['user_graph_agg_%d' % (i+1) + 'output_weight']
                cur_output_user = np.delete(prev_output_user, useless + model.input_dim, axis=0)
                params['user_graph_agg_%d' % (i+1) + 'output_weight'] = cur_output_user

                # item
                prev_output_item = params['item_graph_agg_%d' % i + 'output_weight']
                useless = []
                for j in range(prev_output_item.shape[1] - model.ex_k, prev_output_item.shape[1]):
                    if np.count_nonzero(prev_output_item[:, j]) == 0:
                        useless.append(j)
                useless = np.array(useless)
                cur_output_item = np.delete(prev_output_item, useless, axis=1)
                params['item_graph_agg_%d' % i + 'output_weight'] = cur_output_item
                prev_output_item = params['item_graph_agg_%d' % (i+1) + 'output_weight']
                cur_output_item = np.delete(prev_output_item, useless + model.input_dim, axis=0)
                params['item_graph_agg_%d' % (i+1) + 'output_weight'] = cur_output_item

            self.destroy_graph()
            model.load_params(params)
            model.build_model(prediction=True)

            best_valid_recall20, best_test_recall20, best_test_recall20, best_test_ndcg20 = self.train_model(model, final=True)

            return best_valid_recall20, best_test_recall20, best_valid_ndcg20, best_test_ndcg20



    def destroy_graph(self):
        tf.reset_default_graph()



