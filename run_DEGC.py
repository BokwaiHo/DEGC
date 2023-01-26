"""
Implementation of our 'Dynamically Expandable Graph Convolutions'(DEGC) algorithm for streaming recommendation setting.
"""
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import argparse
import datetime
import time
import pickle
from run_baselines_segments import *
import config_degc
from log2 import Logger
from sampler import pad_adj, WarpSampler
from data_utils.stream_data_generator import Stream_Data
from data_utils.utils import *
from data_utils.reservoir_util import *
from data_utils.preprocessing import generate_index_dict, convert_dict_to_list
from metrics import *
from scipy.stats import wasserstein_distance
from models.mgccf import MGCCF as MGCCF
from models.ngcf import NGCF as NGCF
from models.lightgcn import LightGCN as LightGCN
from sklearn.neighbors import kneighbors_graph
from degc.DEGC import DEGC

import random


class DEGC_Rec:
    def __init__(self, args):
        self.args = args
        self.epochs = args.num_epoch
        self.batch_pairs = args.batch_pairs
        self.lr = args.learning_rate
        self.max_trials = args.max_trials
        self.penalty = args.penalty
        self.base_model = args.base_model
        self.embedded_dimension = eval(self.args.embedded_dimension)
        self.args.load_save_path_prefix = self.args.load_save_path_prefix + self.args.base_model + '/'
        self.LOG_SAVE_PATH_PREFIX = self.args.load_save_path_prefix
        self.save_ckpt = self.LOG_SAVE_PATH_PREFIX + args.log_folder + '/' + args.save_cp + '.ckpt' if args.save_cp else ''
        self.load_ckpt = self.LOG_SAVE_PATH_PREFIX + args.log_folder + '/' + args.load_cp + '.ckpt' if args.load_cp else ''
        self.saved_ckpt = []
        #loading data
        self.n_segments = args.last_segment_time - args.first_segment_time + 1
        self.stream_data_generator = Stream_Data(dataset=args.dataset, first_segment_time = args.first_segment_time, last_segment_time = args.last_segment_time, shuffle=False, test_ratio = args.test_ratio, \
                                 valid_test_ratio=args.valid_test_ratio, seed=args.seed)
        self.data_segments = self.stream_data_generator.segments
        """
        if args.base_model == 'MGCCF':
            self.evaluater = evaluater_MGCCF(self.args)
        elif args.base_model == 'NGCF':
            self.evaluater = evaluater_NGCF(self.args)
        else:
            NotImplementedError
        """

    def create_session(self):
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        return sess

    def train(self):
        save_ratio = ''.join([str(s) for s in eval(self.args.con_ratios)])
        save_lambda_con = '-'.join(str(s) for s in eval(self.args.lambda_contrastive))
        save_layer_dim = ''.join(str(s) for s in eval(self.args.embedded_dimension))
        save_k = ''.join(str(s) for s in eval(self.args.k_centroids))
        save_setting = self.args.algorithm + f"cp{self.args.ui_con_positive}cr{save_ratio}lc{save_lambda_con}" + \
                   f"K{save_k}lr{self.args.learning_rate}" + f"bs{self.args.batch_pairs}ld{save_layer_dim}" + (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%b_%d_%H_%M_%S")
        
        #Initiliazer Controller
        #controller = Controller(self.args)
        self.result_process = dict()
        self.result_process['recall20_val'] = []
        self.result_process['ndcg20_val'] = []
        self.best_params = dict()

        for segment in range(self.n_segments):
            self.best_params[segment] = [0,0,0]
            if segment == 0:
                if self.args.log_name:
                    now = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%b_%d_%H_%M_%S") + '-segment' + str(segment)
                    name = self.args.log_folder + '/' + self.args.log_name + '-' + self.args.base_model + '-' + self.args.dataset
                    log_save_path = self.LOG_SAVE_PATH_PREFIX + name + '/' + now
                    result_log_name = self.args.log_files + '/' +  self.args.base_model + '/' + self.args.dataset + '/' + save_setting
                    logger = Logger(result_log_name, name, now, self.args.load_save_path_prefix)
                    logger.open(result_log_name + f'/log.train_segment_{segment}.txt', mode='a')
                    for arg in vars(self.args):
                        logger.write(arg + '=' + str(getattr(self.args, arg)) + '\n')

                else:
                    logger = None
                
                self.save_ckpt = log_save_path + 'model.ckpt'
                self.saved_ckpt.append(self.save_ckpt)

                train_n_user, train_n_item = self.data_segments[segment]['n_user_train'], self.data_segments[segment]['n_item_train']
                val_n_user, val_n_item = self.data_segments[segment]['n_user_val'], self.data_segments[segment]['n_item_val']
                test_n_user, test_n_item = self.data_segments[segment]['n_user_test'], self.data_segments[segment]['n_item_test']

                train_set = self.data_segments[segment]['train']
                val_set = self.data_segments[segment]['val']
                test_set = self.data_segments[segment]['test']

                train_matrix = self.data_segments[segment]['train_matrix']
                val_matrix = self.data_segments[segment]['val_matrix']
                test_matrix = self.data_segments[segment]['test_matrix']

                graph_path = [self.args.graph_path + self.args.dataset + '/' + 'uu_graph_0.npy', \
                              self.args.graph_path + self.args.dataset + '/' + 'ii_graph_0.npy']  

                print('self.save_ckpt', self.save_ckpt)   

                self.DEGC = DEGC(self.args,
                                        segment,
                                        [train_set, train_n_user, train_n_item, train_matrix],
                                        [val_set, val_n_user, val_n_item, val_matrix],
                                        [test_set, test_n_user, test_n_item, test_matrix],
                                        None,
                                        None,
                                        self.epochs,
                                        logger=logger,
                                        load_checkpoint = self.load_ckpt,
                                        save_checkpoint = self.save_ckpt,
                                        graph_path = graph_path                                     
                )
                best_valid_recall20, best_test_recall20, best_valid_ndcg20, best_test_ndcg20 = self.DEGC.DEGC_main()
                self.load_ckpt = self.save_ckpt

                print('recall20_test is ', best_test_recall20)
                print('ndcg20_test is ', best_test_ndcg20)
                logger.write('recall20_test is ' + str(best_test_recall20) + '\n')
                logger.write('ndcg20_test is ' + str(best_test_ndcg20) + '\n') 

            else:
                if self.args.log_name:
                    now = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%b_%d_%H_%M_%S") + '-segment' + str(segment)
                    name = self.args.log_folder + '/' + self.args.log_name + '-' + self.args.base_model + '-' + self.args.dataset
                    log_save_path = self.LOG_SAVE_PATH_PREFIX + name + '/' + now
                    result_log_name = self.args.log_files + '/' + self.args.base_model + '/' + self.args.dataset + '/' + save_setting
                    logger = Logger(result_log_name, name, now, self.args.load_save_path_prefix)
                    logger.open(result_log_name + f'/log.train_segment_{segment}.txt', mode='a')
                    for arg in vars(self.args):
                        logger.write(arg + '=' + str(getattr(self.args, arg)) + '\n')

                else:
                    logger = None

                best_reward = 0
                recall20_val_list = []
                ndcg20_val_list = []


                self.save_ckpt = log_save_path + '/model.ckpt'
                self.saved_ckpt.append(self.save_ckpt)
                
                next_segment = segment + 1
                prev_segment = segment - 1

                train_set = self.data_segments[segment]['train']
                n_user_train, n_item_train = self.data_segments[segment]['n_user_train'], self.data_segments[segment]['n_item_train']
                n_old_user_train, n_old_item_train = self.data_segments[prev_segment]['n_user_train'], self.data_segments[prev_segment]['n_item_train']
                cur_train_matrix = self.data_segments[segment]['train_matrix']
                prev_train_set = self.data_segments[prev_segment]['train']
                prev_train_matrix = self.data_segments[prev_segment]['train_matrix']

                val_set = self.data_segments[segment]['val']
                n_item_val, n_user_val = self.data_segments[segment]['n_item_val'], len(val_set)
                cur_val_matrix = self.data_segments[segment]['val_matrix']
                test_set = self.data_segments[segment]['test']
                n_item_test, n_user_test = self.data_segments[segment]['n_item_test'], len(test_set)
                cur_test_matrix = self.data_segments[segment]['test_matrix']

                full_batch_append = ''
                node_deg_delta = None
                if self.base_model == 'MGCCF':
                    graph_path = [
                        self.args.graph_path + self.args.dataset + '/' + 'uu_graph_' + str(segment) + full_batch_append + '.pkl', \
                        self.args.graph_path + self.args.dataset + '/' + 'ii_graph_' + str(segment) + full_batch_append + '.pkl']
                    prev_graph_path = [
                        self.args.graph_path + self.args.dataset + '/' + 'uu_graph_' + str(prev_segment) + full_batch_append + '.pkl', \
                        self.args.graph_path + self.args.dataset + '/' + 'ii_graph_' + str(prev_segment) + full_batch_append + '.pkl']
                
                self.DEGC = DEGC(self.args,
                                        segment,
                                        [train_set, n_user_train, n_item_train, cur_train_matrix],
                                        [val_set, n_user_val, n_item_val, cur_val_matrix],
                                        [test_set, n_user_test, n_item_test, cur_test_matrix],   
                                        prev_train_set,
                                        prev_train_matrix,      
                                        self.epochs,
                                        n_old_user=n_old_user_train,
                                        n_old_item=n_old_item_train,
                                        node_deg_delta=node_deg_delta,
                                        logger=logger,
                                        load_checkpoint=self.load_ckpt,
                                        save_checkpoint=self.save_ckpt,
                                        graph_path=graph_path
                )

                best_valid_recall20, best_test_recall20, best_valid_ndcg20, best_test_ndcg20 = self.DEGC.DEGC_main()



                self.load_ckpt = self.save_ckpt

                print('recall20_test is ', best_test_recall20)
                print('ndcg20_test is ', best_test_ndcg20)
                logger.write('recall20_test is ' + str(best_test_recall20) + '\n')
                logger.write('ndcg20_test is ' + str(best_test_ndcg20) + '\n') 

        for ckpt in self.saved_ckpt:
            os.system('rm ' + ckpt + '.*')
        

            
            
    def train_old(self):
        self.best_params={}
        self.result_process = []
        for task_id in range(0,self.num_tasks):
            self.best_params[task_id] = [0,0]
            if task_id == 0:
                with tf.Graph().as_default() as g:
                    with tf.name_scope("before"):
                        inputs = tf.placeholder(shape=(None, 784), dtype=tf.float32)
                        y = tf.placeholder(shape=(None, 10), dtype=tf.float32)
                        w1 = tf.Variable(tf.truncated_normal(shape=(784,312), stddev=0.01))
                        b1 = tf.Variable(tf.constant(0.1, shape=(312,)))
                        w2 = tf.Variable(tf.truncated_normal(shape=(312,128), stddev=0.01))
                        b2 = tf.Variable(tf.constant(0.1, shape=(128,)))
                        w3 = tf.Variable(tf.truncated_normal(shape=(128,10), stddev=0.01))
                        b3 = tf.Variable(tf.constant(0.1, shape=(10,)))
                        output1 = tf.nn.relu(tf.nn.xw_plus_b(inputs,w1,b1,name="output1"))
                        output2 = tf.nn.relu(tf.nn.xw_plus_b(output1,w2,b2,name="output2"))
                        output3 = tf.nn.xw_plus_b(output2,w3,b3,name="output3")
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output3)) + \
                               0.0001*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))
                        if self.args.optimizer=="adam":
                            optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
                        elif self.args.optimizer=="rmsprop":
                            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
                        elif self.args.optimizer=="sgd":
                            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                        else:
                            raise Exception("please choose one optimizer")
                        train_step = optimizer.minimize(loss)
                        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(output3,axis=1)),tf.float32))
                        sess = self.create_session()
                        sess.run(tf.global_variables_initializer())
                        l = len(self.task_list[0][1])
                        for epoch in range(self.epochs):
                            flag = 0
                            for _ in range(l//self.batch_size+1):
                                batch_xs, batch_ys = (self.task_list[task_id][0][flag:flag+self.batch_size],self.task_list[task_id][1][flag:flag+self.batch_size])
                                flag += self.batch_size
                                sess.run(train_step,feed_dict={inputs:batch_xs, y:batch_ys})
                        accuracy_test = sess.run(accuracy, feed_dict={inputs:self.task_list[task_id][4], y:self.task_list[task_id][5]})
                        print("test accuracy: ", accuracy_test)
                        self.vars = sess.run([w1,b1,w2,b2,w3,b3])
                    self.best_params[task_id] = [accuracy_test,self.vars]
            else:
                tf.reset_default_graph()
                controller = Controller(self.args)
                results = []
                best_reward = 0
                for trial in range(self.max_trials):
                    actions = controller.get_actions()
                    print("***************actions*************",actions)
                    accuracy_val, accuracy_test = self.evaluates.evaluate_action(var_list = self.vars, 
                             actions=actions, task_id = task_id)

                    results.append(accuracy_val)
                    print("test accuracy: ", accuracy_test)
                    reward = accuracy_val - self.penalty*sum(actions)
                    print("reward: ", reward)
                    if reward > best_reward:
                        best_reward = reward
                        self.best_params[task_id] = (accuracy_test, self.evaluates.var_list)
                    controller.train_controller(reward)
                controller.close_session()
                self.result_process.append(results)
                self.vars = self.best_params[task_id][1]
        
if __name__ == "__main__":

    #parse arguments
    args = config_degc.parse_arguments()
    print('using GPU', str(args.device))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    #seed
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
    
    #streaming recommendation process
    DEGC_Rec = DEGC_Rec(args)
    DEGC_Rec.train()  



