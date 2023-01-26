from mailbox import NotEmptyError
from multiprocessing.spawn import old_main_modules
import os, time, datetime
from re import L
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cProfile, pstats
import tensorflow as tf
import numpy as np
from scipy.special import softmax

import config_baselines
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
#from rcgl.evaluate import evaluate
#from rcgl.policy_gradient import Controller
import random

def write_prediction_to_logger(logger, precision, recall, MAP, ndcg, epoch, name):
    if logger is not None:
        logger.write('Epoch: {} ({}) \n'.format(epoch, name))
        logger.write('precision :' + str(precision) + '\n')
        logger.write('recall :' + str(recall) + '\n')
        logger.write('MAP :' + str(MAP) + '\n')
        logger.write('ndcg :' + str(ndcg) + '\n')
    else:
        print('Epoch: {} ({}) \n'.format(epoch, name))
        print('precision :' + str(precision))
        print('recall :' + str(recall))
        print('MAP :' + str(MAP))
        print('ndcg :' + str(ndcg))

def convert_list_to_adj(adj, n_item):
    adj_mat = np.zeros((len(adj), n_item))
    for user, idx in enumerate(adj):
        adj_mat[user][idx] = 1
    return adj_mat

def convert_adj_to_cat(old_list, n_item, n_cluster, y_kmeans):
    old_adj = convert_list_to_adj(old_list, n_item)
    new_adj = np.zeros((old_adj.shape[0], n_cluster))
    for i in range(old_adj.shape[0]):
        for j in range(len(y_kmeans)):
            new_adj[i][y_kmeans[j]] += old_adj[i][j]
    return new_adj

def convert_adj_cat_mean(old_list, emb, n_cluster, y_kmeans):
    new_adj = np.zeros((len(old_list), n_cluster, emb.shape[1]))
    for i in range(new_adj.shape[0]):
        neighs = old_list[i]
        for j in range(n_cluster):
            if len(np.where(y_kmeans[neighs] == j)[0]) > 0:
                new_adj[i][j] == np.mean(emb[np.where(y_kmeans[neighs] == j)[0]], axis=0)
    return new_adj

def generate_local_neighbors(u_adj_list, u_emb, i_emb, distance_mode, num_neigh):
    u_ls_matrix = np.zeros([u_emb.shape[0], num_neigh])
    u_ls_index = np.zeros([u_emb.shape[0], num_neigh])
    for u, i_list in u_adj_list.items():
        if u >= u_emb.shape[0]:
            break
        i_list = [x for x in i_list if x < i_emb.shape[0]]
        if len(i_list) > 0:
            #i_list = [x for x in i_list if x < i_emb.shape[0]]

            #Sample Version
            if len(i_list) > 10:
                #sampling without replacement
                i_list = random.sample(i_list, 10)
            else:
                #sampling with replacement when setting parameter 'replace' as default(True)
                print('i_list', i_list)
                i_list = random.choices(i_list, k=10)
            u_1hop_emb = np.take(i_emb, i_list, axis=0)
            if distance_mode == 'enu':
                u_i_distance = np.square(np.linalg.norm(u_1hop_emb - u_emb[u], axis=1))
            elif distance_mode == 'inner_product':
                u_i_distance = np.sum(u_1hop_emb * u_emb[u], axis=1)
            elif distance_mode == 'poly':
                u_i_distance = np.square(np.sum(u_1hop_emb * u_emb[u], axis=1))
            elif distance_mode == 'rbf':
                u_i_distance = np.square(np.linalg.norm(u_1hop_emb - u_emb[u], axis=1))
                u_i_distance = np.exp(-0.5 * u_i_distance)
            else:
                NotImplementedError
            u_ls = softmax(u_i_distance)
            u_ls_matrix[u] = u_ls
            u_ls_index[u] = i_list
        else:
            i_list = random.sample(range(i_emb.shape[0]), num_neigh)
            u_ls_index[u] = i_list
            u_i_distance = np.ones(num_neigh) / num_neigh
            u_ls_matrix[u] = u_i_distance

    return u_ls_matrix, u_ls_index


def get_local_structure(u_adj_list, i_adj_list, u_emb, i_emb, distance_mode, num_neigh=10):
    assert distance_mode != ''
    u_ls_matrix, u_ls_index = generate_local_neighbors(u_adj_list, u_emb, i_emb, distance_mode, num_neigh)
    i_ls_matrix, i_ls_index = generate_local_neighbors(i_adj_list, i_emb, u_emb, distance_mode, num_neigh)
    
    return u_ls_matrix, i_ls_matrix, u_ls_index, i_ls_index

def load_self_neighbours(file_path, data_group, n_rows, n_neighbours, graph_adj_matrix, n_negative=None, include_self=False):
    graph_file_path = file_path[:-4] + f'_{n_neighbours}_' + data_group + file_path[-4:]
    if os.path.isfile(graph_file_path):
        self_neigh_graph = load_pickle(graph_file_path, '')
    else:
        self_neigh_graph = kneighbors_graph(graph_adj_matrix, n_neighbours, mode='distance', metric='cosine', include_self = include_self)
        save_pickle(self_neigh_graph, graph_file_path[:-4], '')
    self_neighs = self_neigh_graph.tocoo().col
    self_neighs = np.array(np.array_split(self_neighs, n_rows))
    if n_negative is not None:
        neigh_pairs = []
        for i in range(len(self_neighs)):
            neigh_pairs += [[i, j] for j in self_neighs[i]]
        neigh_pairs = np.array(neigh_pairs)
        user_to_positive_set = {u: set(row) for u, row in enumerate(self_neighs)}

        #sample negative samples
        negative_samples = np.random.randint(0, len(self_neighs), size=(len(self_neighs), n_negative))
        for negatives, i in zip(negative_samples, range(len(negative_samples))):
            for j, neg in enumerate(negatives):
                while neg in user_to_positive_set[i]:
                    negative_samples[i, j] = neg = np.random.randint(0, len(self_neighs))
    else:
        negative_samples = None
    return self_neighs, negative_samples

def load_self_neighboursplus(file_path, data_group, n_rows, n_neighbours, graph_adj_matrix, n_negative=None, include_self=False):
    if include_self:
        flag = '_wself'
    else:
        flag = '_woself'
    graph_file_path = file_path[:-4] + f'_{n_neighbours}_' + data_group + flag + file_path[-4:]
    if os.path.isfile(graph_file_path):
        self_neigh_graph = load_pickle(graph_file_path, '')
    else:
        self_neigh_graph = kneighbors_graph(graph_adj_matrix, n_neighbours, mode='distance', metric='cosine', include_self = include_self)
        save_pickle(self_neigh_graph, graph_file_path[:-4], '')
    self_neighs = self_neigh_graph.tocoo().col
    self_neighs = np.array(np.array_split(self_neighs, n_rows))
    if n_negative is not None:
        neigh_pairs = []
        for i in range(len(self_neighs)):
            neigh_pairs += [[i, j] for j in self_neighs[i]]
        neigh_pairs = np.array(neigh_pairs)
        user_to_positive_set = {u: set(row) for u, row in enumerate(self_neighs)}

        #sample negative samples
        negative_samples = np.random.randint(0, len(self_neighs), size=(len(self_neighs), n_negative))
        for negatives, i in zip(negative_samples, range(len(negative_samples))):
            for j, neg in enumerate(negatives):
                while neg in user_to_positive_set[i]:
                    negative_samples[i, j] = neg = np.random.randint(0, len(self_neighs))
    else:
        negative_samples = None
    return self_neighs, negative_samples

def load_bi_neighbours(adj_mat, u_num_neigh, i_num_neigh, u_self_n_negative, i_self_n_negative):
    u_adj_dict, i_adj_dict = sparse_adj_matrix_to_dicts(adj_mat)
    u_pos_neighs, i_pos_neighs = pad_adj(u_adj_dict, u_num_neigh, adj_mat.shape[1]), \
                                 pad_adj(u_adj_dict, u_num_neigh, adj_mat.shape[0])
    adj_mat = np.array(adj_mat.todense()).astype(np.float64)
    u_neg_neighs = []
    for i in range(adj_mat.shape[0]):
        neg_idx = list(np.where(adj_mat[i]==0)[0])
        u_neg_neighs.append(random.sample(neg_idx, u_self_n_negative))
    i_neg_neighs = []
    for i in range(adj_mat.shape[1]):
        neg_idx = list(np.where(adj_mat.T[i]==0)[0])
        i_neg_neighs.append(random.sample(neg_idx, i_self_n_negative))
    return u_pos_neighs, i_pos_neighs, u_neg_neighs, i_neg_neighs    


# Transductive: train node-train node edge, train node-valid node edge, train node-test node edge
#               cannot infer on the nodes that have not been seen in the training process
# Inductive: train node-train node edge only
#            can infer on the nodes that have not been seen in the training process (infer the their embedding via aggregation operation)
def train_model(parser, segment, train_info, val_info, test_info, old_train_set, old_train_matrix, \
                n_epoch, n_old_user=0, n_old_item=0, node_deg_delta=None, logger=None, load_checkpoint='',
                save_checkpoint='', graph_path=None):
    # The save_checkpoint should be set as empty when running the validation experiments and collecting data for RL controller update at each timestep
    base_model = parser.base_model
    train_set, n_user_train, n_item_train, train_matrix = train_info
    val_set, n_user_val, n_item_val, val_matrix = val_info
    test_set, n_user_test, n_item_test, test_matrix = test_info

    save_ckpt = save_checkpoint
    
    if base_model == 'MGCCF':
        user_self_neighs, _ = load_self_neighbours(graph_path[0], 'train', n_user_train, parser.num_neigh, train_matrix)
        item_self_neighs, _ = load_self_neighbours(graph_path[1], 'train', n_item_train, parser.num_neigh, train_matrix.transpose())
        user_self_neighs_val, _ = load_self_neighbours(graph_path[0], 'val', n_user_val, parser.num_neigh, val_matrix)
        item_self_neighs_val, _ = load_self_neighbours(graph_path[1], 'val', n_item_val, parser.num_neigh, val_matrix.transpose())
        user_self_neighs_test, _ = load_self_neighbours(graph_path[0], 'test', n_user_test, parser.num_neigh, test_matrix)
        item_self_neighs_test, _ = load_self_neighbours(graph_path[1], 'test', n_item_test, parser.num_neigh, test_matrix.transpose())

    #prepare train data
    u_adj_dict_train, i_adj_dict_train = sparse_adj_matrix_to_dicts(train_matrix)
    u_adj_list_train, i_adj_list_train = pad_adj(u_adj_dict_train, parser.max_degree, n_item_train), pad_adj(i_adj_dict_train, parser.max_degree, n_user_train)

    u_adj_dict_val, i_adj_dict_val = sparse_adj_matrix_to_dicts(val_matrix)
    u_adj_list_val, i_adj_list_val = pad_adj(u_adj_dict_val, parser.max_degree, n_item_val), pad_adj(i_adj_dict_val, parser.max_degree, n_user_val)

    u_adj_dict_test, i_adj_dict_test = sparse_adj_matrix_to_dicts(test_matrix)
    u_adj_list_test, i_adj_list_test = pad_adj(u_adj_dict_test, parser.max_degree, n_item_test), pad_adj(i_adj_dict_test, parser.max_degree, n_user_test)
    
    print('base model', base_model)
    if base_model == 'MGCCF':
        if segment > 0:
            model = MGCCF(load_checkpoint,
                        [eval(parser.embedded_dimension)[0], n_user_train, n_item_train],
                        eval(parser.embedded_dimension)[1:],
                        parser.max_degree,
                        eval(parser.gcn_sample),
                        ['adam', parser.learning_rate, parser.epsilon],
                        'my_mean',
                        parser.activation,
                        parser.neighbor_dropout,
                        parser.l2,
                        parser.dist_embed,
                        parser.num_self_neigh,
                        parser.num_neg,
                        parser.ui_con_positive,
                        eval(parser.con_ratios),
                        inc_reg = [parser.lambda_mse, parser.lambda_distillation, parser.lambda_global_distill],
                        old_num_user = n_old_user,
                        old_num_item = n_old_item,
                        distill_mode = parser.distill_mode,
                        k_centroids=eval(parser.k_centroids),
                        tau = parser.tau,
                        num_neigh = parser.num_neigh,
                        local_distill_mode = parser.local_mode,
                        contrastive_mode=parser.contrastive_mode,
                        layer_wise=parser.layer_wise,
                        layer_l2_mode=parser.layer_l2_mode,
                        lambda_layer_l2=parser.lambda_layer_l2,
                        lambda_contrastive=eval(parser.lambda_contrastive))
        else:
            model = MGCCF(load_checkpoint,
                        [eval(parser.embedded_dimension)[0], n_user_train, n_item_train],
                        eval(parser.embedded_dimension)[1:],
                        parser.max_degree,
                        eval(parser.gcn_sample),
                        ['adam', parser.learning_rate, parser.epsilon],
                        'my_mean',
                        parser.activation,
                        parser.neighbor_dropout,
                        parser.l2,
                        dist_embed=parser.dist_embed,
                        num_self_neigh=parser.num_self_neigh,
                        neg_item_num=parser.num_neg,
                        ui_con_positive=parser.ui_con_positive,
                        con_ratios=eval(parser.con_ratios),
                        old_num_user = n_old_user,
                        old_num_item = n_old_item,
                        num_neigh = parser.num_neigh,
                        layer_wise=0,
                        layer_l2_mode=0,
                        lambda_layer_l2='[0,0,0]') 
    elif  base_model == 'NGCF':
        if segment > 0:
            model = NGCF(load_checkpoint,
                        [eval(parser.embedded_dimension)[0], n_user_train, n_item_train],
                        eval(parser.embedded_dimension)[1:],
                        parser.max_degree,
                        eval(parser.gcn_sample),
                        ['adam', parser.learning_rate, parser.epsilon],
                        'my_mean',
                        parser.activation,
                        parser.neighbor_dropout,
                        parser.l2,
                        parser.num_neg,
                        parser.ui_con_positive,
                        eval(parser.con_ratios),
                        inc_reg = [parser.lambda_mse, parser.lambda_distillation, parser.lambda_global_distill],
                        old_num_user = n_old_user,
                        old_num_item = n_old_item,
                        distill_mode = parser.distill_mode,
                        k_centroids=eval(parser.k_centroids),
                        tau = parser.tau,
                        local_distill_mode = parser.local_mode,
                        contrastive_mode=parser.contrastive_mode,
                        layer_wise=parser.layer_wise,
                        layer_l2_mode=parser.layer_l2_mode,
                        lambda_layer_l2=parser.lambda_layer_l2,
                        lambda_contrastive=eval(parser.lambda_contrastive))
        else:
            model = NGCF(load_checkpoint,
                        [eval(parser.embedded_dimension)[0], n_user_train, n_item_train],
                        eval(parser.embedded_dimension)[1:],
                        parser.max_degree,
                        eval(parser.gcn_sample),
                        ['adam', parser.learning_rate, parser.epsilon],
                        'my_mean',
                        parser.activation,
                        parser.neighbor_dropout,
                        parser.l2,
                        neg_item_num=parser.num_neg,
                        ui_con_positive=parser.ui_con_positive,
                        con_ratios=eval(parser.con_ratios),
                        old_num_user = n_old_user,
                        old_num_item = n_old_item,
                        layer_wise=0,
                        layer_l2_mode=0,
                        lambda_layer_l2='[0,0,0]')    
    elif  base_model == 'LightGCN':
        if segment > 0:
            model = LightGCN(load_checkpoint,
                       [eval(parser.embedded_dimension)[0], n_user_train, n_item_train],
                        eval(parser.embedded_dimension)[1:],
                        parser.max_degree,
                        eval(parser.gcn_sample),
                        ['adam', parser.learning_rate, parser.epsilon],
                        parser.batch_pairs,
                        parser.neighbor_dropout,
                        parser.l2,
                        parser.num_neg,
                        parser.ui_con_positive,
                        eval(parser.con_ratios),
                        inc_reg = [parser.lambda_mse, parser.lambda_distillation, parser.lambda_global_distill],
                        old_num_user = n_old_user,
                        old_num_item = n_old_item,
                        distill_mode = parser.distill_mode,
                        k_centroids=eval(parser.k_centroids),
                        tau = parser.tau,
                        local_distill_mode = parser.local_mode,
                        contrastive_mode=parser.contrastive_mode,
                        layer_wise=parser.layer_wise,
                        layer_l2_mode=parser.layer_l2_mode,
                        lambda_layer_l2=parser.lambda_layer_l2,
                        lambda_contrastive=eval(parser.lambda_contrastive))
        else:
            model = LightGCN(load_checkpoint,
                        [eval(parser.embedded_dimension)[0], n_user_train, n_item_train],
                        eval(parser.embedded_dimension)[1:],
                        parser.max_degree,
                        eval(parser.gcn_sample),
                        ['adam', parser.learning_rate, parser.epsilon],
                        parser.batch_pairs,
                        parser.neighbor_dropout,
                        parser.l2,
                        neg_item_num=parser.num_neg,
                        ui_con_positive=parser.ui_con_positive,
                        con_ratios=eval(parser.con_ratios),
                        old_num_user = n_old_user,
                        old_num_item = n_old_item,
                        layer_wise=0,
                        layer_l2_mode=0,
                        lambda_layer_l2='[0,0,0]')   
    else:
        NotImplementedError

    num_pairs = 0
    for i in range(len(train_set)):
        num_pairs += len(train_set[i]) 
    num_iter = int(num_pairs / parser.batch_pairs) + 1
    iter_time = []

    sampler = WarpSampler(train_set,
                          n_item_train,
                          batch_size = parser.batch_pairs,
                          n_negative = parser.num_neg,
                          n_workers = 2,
                          check_negative = True
                          )
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess_config = tf.compat.v1.ConfigProto()
    # consolidate fragemented idle GPU memory segments for reallocating to dynamic computation graphs
    #sess_config.gpu_options.allow_growth = True 

    with model.graph.as_default():
        saver = tf.train.Saver(max_to_keep=100)
        new_var_list = [x for x in saver._var_list if ("embedding" not in x.name and "Adam" not in x.name and "input" not in x.name)]
        saver_2 = tf.train.Saver(var_list=new_var_list)

    with tf.compat.v1.Session(graph=model.graph, config=sess_config) as sess:
        #initialize variables
        sess.run(tf.global_variables_initializer())
        #load checkpoints
        u_emb_val = sess.run(model.user_embeddings)
        i_emb_val = sess.run(model.item_embeddings)

        #loaad existing model
        if load_checkpoint != "":
            saver_2.restore(sess, load_checkpoint)
            #only load existing nodes'embedding
            old_u_emb_val = tf.train.load_variable(load_checkpoint, 'model/user_embedding')
            old_i_emb_val = tf.train.load_variable(load_checkpoint, 'model/item_embedding')

            #If not first segment, u_emb_val would be set as old_embedding + average of similar old user embedding
            #for new users, so for items.
            u_emb_val[:old_u_emb_val.shape[0], ] = old_u_emb_val
            i_emb_val[:old_i_emb_val.shape[0], ] = old_i_emb_val


            # initialize new node as mean of 2-hop neighbors
            if parser.new_node_init == '2hop_mean':
                if n_user_train > n_old_user:
                    new_users_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('user', old_u_emb_val, train_matrix)
                    u_emb_val[n_old_user:,] = new_users_init
                if n_item_train > n_old_item:
                    new_items_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('item', old_i_emb_val, train_matrix)
                    i_emb_val[n_old_item:, ] = new_items_init

            #teacher's 1-hop local structure
            if parser.lambda_distillation > 0 and parser.local_mode == 'LSP_s':
                u_ls_matrix, i_ls_matrix, u_ls_index, i_ls_index = get_local_structure(u_adj_dict_train, i_adj_dict_train, old_u_emb_val, old_i_emb_val, parser.distill_mode)

            if parser.lambda_distillation > 0 and parser.local_mode == 'local_anchor':
                time_info.append(('start calc local_anchor coef', time.time()))
                #simple one-hop
                sub_matrix = train_matrix[:old_u_emb_val.shape[0], :old_i_emb_val.shape[0]]
                sub_matrix_dense = np.array(sub_matrix.todense()).astype(np.float64)

                u_1hop_means = np.zeros(old_u_emb_val.shape)
                for u in range(len(u_1hop_means)):
                    u_idx = sub_matrix_dense[u].nonzero()
                    u_1hop_means[u] = np.mean(np.take(old_i_emb_val, u_idx[0], axis=0), axis=0)
                u_i_prod = np.sum(old_u_emb_val * u_1hop_means, axis=1)

                i_1hop_means = np.zeros(old_i_emb_val.shape)
                for i in range(len(i_1hop_means)):
                    i_udx = (sub_matrix_dense.T)[i].nonzero()
                    i_1hop_means[i] = np.mean(np.take(old_u_emb_val, i_udx[0], axis=0), axis=0)
                i_u_prod = np.sum(old_i_emb_val * i_1hop_means, axis=1)

                u_i_adj_idx = np.stack(np.nonzero(sub_matrix_dense), axis=-1)
                i_u_adj_idx = np.stack(np.nonzero(sub_matrix_dense.T), axis=-1)
                time_info.append(('finish calc local_anchor coef', time.time()))


            if parser.layer_wise:
                old_user_medium_input_1 = tf.train.load_variable(load_checkpoint, 'model/user_medium_input_1')
                old_user_medium_input_2 = tf.train.load_variable(load_checkpoint, 'model/user_medium_input_2')
                old_item_medium_input_1 = tf.train.load_variable(load_checkpoint, 'model/item_medium_input_1')
                old_item_medium_input_2 = tf.train.load_variable(load_checkpoint, 'model/item_medium_input_2')


            if parser.contrastive_mode:
                ui_pos_neighs, iu_pos_neighs, ui_neg_neighs, iu_neg_neighs = \
                    load_bi_neighbours(old_train_matrix[:old_u_emb_val.shape[0], :old_i_emb_val.shape[0]],
                                       parser.ui_con_positive, int(parser.ui_con_positive*eval(parser.con_ratios)[1]),
                                       int(parser.ui_con_positive*eval(parser.con_ratios)[0]), int(parser.ui_con_positive*eval(parser.con_ratios)[2]))
                
                if parser.contrastive_mode == "Multi":
                    uu_pos_neighs, uu_neg_neighs = \
                        load_self_neighboursplus(parser.graph_path, 'adaptive', old_u_emb_val.shape[0], int(parser.ui_con_positive*eval(parser.con_ratios)[3]),
                                                 old_train_matrix[:old_u_emb_val.shape[0], :old_i_emb_val.shape[0]],
                                                 int(parser.ui_con_positive*eval(parser.con_ratios)[4]), parser.include_self)

                    ii_pos_neighs, ii_neg_neighs = \
                        load_self_neighboursplus(parser.graph_path, 'adaptive', old_i_emb_val.shape[0], int(parser.ui_con_positive*eval(parser.con_ratios)[5]),
                                                 old_train_matrix[:old_u_emb_val.shape[0], :old_i_emb_val.shape[0]].transpose(),
                                                 int(parser.ui_con_positive*eval(parser.con_ratios)[6]), parser.include_self)   

            if parser.lambda_global_distill > 0:
                time_info.append(('start calc global_achor coef', time.time()))
                n_u_anchor, n_i_anchor = eval(parser.k_centroids)

                u_kmeans = KMeans(n_clusters = n_u_anchor, random_state=0, n_jobs=10)
                u_idx = u_kmeans.fit_predict(old_u_emb_val)
                u_cluster_matrix = np.zeros([n_u_anchor, old_u_emb_val.shape[0]])
                u_anchor_points = []
                u_cluster = np.zeros(old_u_emb_val.shape[0])
                for k in range(n_u_anchor):
                    k_idx = np.where(u_idx == k)[0]
                    u_cluster[k_idx] = k
                    u_cluster_matrix[k, k_idx] = 1
                    u_anchor_points.append(np.mean(np.take(old_u_emb_val, k_idx, axis=0), axis=0))

                u_anchor_points = np.array(u_anchor_points)

                i_kmeans = KMeans(n_clusters = n_i_anchor, random_state=0, n_jobs=10)
                i_idx = i_kmeans.fit_predict(old_i_emb_val)
                i_cluster_matrix = np.zeros([n_i_anchor, old_i_emb_val.shape[0]])
                i_anchor_points = []
                i_cluster = np.zeros(old_i_emb_val.shape[0])
                for k in range(n_i_anchor):
                    k_idx = np.where(i_idx == k)[0]
                    i_cluster[k_idx] = k
                    i_cluster_matrix[k, k_idx] = 1
                    i_anchor_points.append(np.mean(np.take(old_i_emb_val, k_idx, axis=0), axis=0))

                i_anchor_points = np.array(i_anchor_points)

                u_cluster_adj_idx = np.stack(np.nonzero(u_cluster_matrix), axis=-1)
                i_cluster_adj_idx = np.stack(np.nonzero(i_cluster_matrix), axis=-1)
                anchor_points = np.concatenate([u_anchor_points, i_anchor_points], axis=0)

                # ===== clusters probability distillation ====== #
                u_gs_matrix = np.zeros([old_u_emb_val.shape[0], anchor_points.shape[0]])
                i_gs_matrix = np.zeros([old_i_emb_val.shape[0], anchor_points.shape[0]])

                for u, u_emb in enumerate(old_u_emb_val):
                    u_gs_matrix[u, :n_u_anchor] = np.sum(u_emb * anchor_points[:n_u_anchor], axis=1) #u on u anchor distribution
                    u_gs_matrix[u, n_u_anchor:] = np.sum(u_emb * anchor_points[n_u_anchor:], axis=1) #u on i anchor distribution
                    u_gs_matrix[u, :n_u_anchor] = softmax(u_gs_matrix[u, :n_u_anchor] / parser.tau)
                    u_gs_matrix[u, n_u_anchor:] = softmax(u_gs_matrix[u, n_u_anchor:] / parser.tau)

                for i, i_emb in enumerate(old_i_emb_val):
                    i_gs_matrix[i, :n_u_anchor] = np.sum(i_emb * anchor_points[:n_u_anchor], axis=1) #i on u anchor distribution
                    i_gs_matrix[i, n_u_anchor:] = np.sum(i_emb * anchor_points[n_u_anchor:], axis=1) #i on i anchor distribution
                    i_gs_matrix[i, :n_u_anchor] = softmax(i_gs_matrix[i, :n_u_anchor] / parser.tau)
                    i_gs_matrix[i, n_u_anchor:] = softmax(i_gs_matrix[i, n_u_anchor:] / parser.tau)

                time_info.append(('finish calc global_anchor coef', time.time()))

            model.user_embeddings.load(u_emb_val, sess)
            model.item_embeddings.load(i_emb_val, sess)                


        _epoch = 0
        best_valid_recall20, best_valid_ndcg20, best_valid_epoch, best_test_recall20, best_test_ndcg20 = 0., 0., 0., 0., 0.
        early_stop_flag = 0
        mse_user_reg, mse_item_reg = None, None

        time_info_training = 0
        time_info_eval = 0
        time_info_sampling = 0

        while _epoch <= n_epoch:
            time_info.append(('start epoch' + str(_epoch) + ' training', time.time()))

            if _epoch % 1 == 0:
                time_info_eval_start = time.time()
                if base_model == 'MGCCF':
                    precision, v_recall, MAP, v_ndcg, _, _ = evaluate_model(sess, base_model, model,  val_info, train_matrix, u_adj_list_val, i_adj_list_val,
                                                                        user_self_neighs_val, item_self_neighs_val, n_batch_users=parser.batch_evaluate)
                elif base_model == 'NGCF':
                    precision, v_recall, MAP, v_ndcg, _, _ = evaluate_model(sess, base_model, model,  val_info, train_matrix, u_adj_list_val, i_adj_list_val,
                                                                        n_batch_users=parser.batch_evaluate)                   
                elif base_model == 'LightGCN':
                    precision, v_recall, MAP, v_ndcg, _, _ = evaluate_model(sess, base_model, model, val_info, train_matrix, u_adj_list_val, i_adj_list_val,
                                                                        n_batch_users=parser.batch_evaluate)

                else:
                    NotImplementedError                
                    

                write_prediction_to_logger(logger, precision, v_recall, MAP, v_ndcg, _epoch, 'validation set')
                
                #v_recall[-1] means k is taken as 20
                if v_recall[-1] >= best_valid_recall20:
                    #accelerate: only check testset when finding best model on validation dataset
                    if base_model == 'MGCCF':
                        precision, t_recall, MAP, t_ndcg, _, _ = evaluate_model(sess, base_model, model, test_info, train_matrix,
                                                                            u_adj_list_test, i_adj_list_test, user_self_neighs_test,
                                                                            item_self_neighs_test, n_batch_users=parser.batch_evaluate)
                    elif base_model == 'NGCF':
                        precision, t_recall, MAP, t_ndcg, _, _ = evaluate_model(sess, base_model, model, test_info, train_matrix,
                                                                            u_adj_list_test, i_adj_list_test, n_batch_users=parser.batch_evaluate)  
                    elif base_model == 'LightGCN':
                        precision, t_recall, MAP, t_ndcg, _, _ = evaluate_model(sess, base_model, model, test_info, train_matrix,
                                                                            u_adj_list_test, i_adj_list_test, n_batch_users=parser.batch_evaluate)
                    else:
                        NotImplementedError                      
                    write_prediction_to_logger(logger, precision, t_recall, MAP, t_ndcg, _epoch, 'test set')

                    early_stop_flag = 0
                    best_valid_recall20 = v_recall[-1]
                    best_valid_ndcg20 = v_ndcg[-1]
                    best_valid_epoch = _epoch
                    best_test_recall20 = t_recall[-1]
                    best_test_ndcg20 = t_ndcg[-1]
                    
                    if save_checkpoint != "":
                        # save embedding for next time segement

                        saver.save(sess, save_checkpoint)
                        #save information for later use
                        if _epoch != 0:
                            if parser.reservoir_selection == 'mse_distillation_score':
                                save_pickle(mse_user_reg, save_ckpt[:-10], 'user_distill_score')
                                save_pickle(mse_item_reg, save_ckpt[:-10], 'item_distill_score')
                
                early_stop_flag += 1
                if early_stop_flag > parser.patience and _epoch > parser.min_epoch:
                    if logger is not None:
                        logger.write('early stop triggered at epoch: ' + str(_epoch) + '\n')
                    else:
                        print('early stop triggered at epoch' + str(_epoch))
                    break

                time_info_eval_end = time.time()
                time_info_eval += time_info_eval_end - time_info_eval_start

            _epoch += 1
            if _epoch > n_epoch:
                break

            for iter in range(0, num_iter):
                time_info_sampling_start = time.time()
                user_pos, neg_samples = sampler.next_batch()

                iter_start = time.time()
                time_info_sampling += iter_start - time_info_sampling_start
                if base_model == 'MGCCF':
                    feed_dict = {model.u_id: user_pos[:, 0],
                                model.pos_item_id: user_pos[:, 1],
                                model.neg_item_id: neg_samples,
                                model.u_adj_info_ph: u_adj_list_train,
                                model.v_adj_info_ph: i_adj_list_train,
                                model.u_u_graph_ph: user_self_neighs,
                                model.v_v_graph_ph: item_self_neighs,
                                model.old_user_embedding: u_emb_val,
                                model.old_item_embedding: i_emb_val}
                elif base_model == 'NGCF':
                    feed_dict = {model.u_id: user_pos[:, 0],
                                model.pos_item_id: user_pos[:, 1],
                                model.neg_item_id: neg_samples,
                                model.u_adj_info_ph: u_adj_list_train,
                                model.v_adj_info_ph: i_adj_list_train,
                                model.old_user_embedding: u_emb_val,
                                model.old_item_embedding: i_emb_val}
                elif base_model == 'LightGCN':
                    feed_dict = {model.u_id: user_pos[:, 0],
                                model.pos_item_id: user_pos[:, 1],
                                model.neg_item_id: neg_samples,
                                model.u_adj_info_ph: u_adj_list_train,
                                model.v_adj_info_ph: i_adj_list_train,
                                model.old_user_embedding: u_emb_val,
                                model.old_item_embedding: i_emb_val}
                else:
                    NotImplementedError
                
                if parser.lambda_mse > 0 and node_deg_delta is not None and segment != 0:
                    feed_dict[model.u_mse_coef] = np.take(node_deg_delta[0], user_pos[:,0])
                    feed_dict[model.i_mse_coef] = np.take(node_deg_delta[1], np.concatenate((user_pos[:, 1], neg_samples.flatten())))
                    feed_dict[model.u_mse_coef_dist_score] = node_deg_delta[0][:n_old_user]
                    feed_dict[model.i_mse_coef_dist_score] = node_deg_delta[1][:n_old_item]

                if parser.lambda_distillation > 0 and segment != 0:
                    if parser.local_mode == 'LSP_s':
                        feed_dict[model.old_user_bl_ls] = u_ls_matrix
                        feed_dict[model.old_item_bl_ls] = i_ls_matrix
                        feed_dict[model.old_user_bl_idx] = u_ls_index
                        feed_dict[model.old_item_bl_idx] = i_ls_index
                    elif parser.local_mode == 'local_anchor':
                        feed_dict[model.ui_dist] = u_i_prod
                        feed_dict[model.iu_dist] = i_u_prod
                        feed_dict[model.old_u_i_adj_mat] = (u_i_adj_idx, u_i_adj_idx[:, 1])
                        feed_dict[model.old_i_u_adj_mat] = (i_u_adj_idx, i_u_adj_idx[:, 1])
                
                if parser.contrastive_mode and segment != 0:
                    feed_dict[model.old_ui_pos_neighs] = ui_pos_neighs
                    feed_dict[model.old_iu_pos_neighs] = iu_pos_neighs
                    feed_dict[model.old_ui_neg_neighs] = ui_neg_neighs
                    feed_dict[model.old_iu_neg_neighs] = iu_neg_neighs

                    if parser.contrastive_mode == 'Multi':
                        feed_dict[model.old_uu_pos_neighs] = uu_pos_neighs
                        feed_dict[model.old_ii_pos_neighs] = ii_pos_neighs
                        feed_dict[model.old_uu_neg_neighs] = uu_neg_neighs
                        feed_dict[model.old_ii_neg_neighs] = ii_neg_neighs

                if parser.layer_wise and segment != 0:
                    feed_dict[model.old_user_medium_input_1] = old_user_medium_input_1
                    feed_dict[model.old_item_medium_input_1] = old_item_medium_input_1
                    feed_dict[model.old_user_medium_input_2] = old_user_medium_input_2
                    feed_dict[model.old_item_medium_input_2] = old_item_medium_input_2

                if parser.lambda_global_distill > 0 and segment != 0:
                    # ===== cluster anchors ======= #
                    feed_dict[model.old_user_embedding] = u_emb_val
                    feed_dict[model.old_item_embedding] = i_emb_val
                    feed_dict[model.old_user_gs] = u_gs_matrix
                    feed_dict[model.old_item_gs] = i_gs_matrix
                    feed_dict[model.old_u_cluster_mat] = (u_cluster_adj_idx, u_cluster_adj_idx[:, 1]) # sparse matrix, n_u_anchor * n_user (old_u_emb_val)
                    feed_dict[model.old_i_cluster_mat] = (i_cluster_adj_idx, i_cluster_adj_idx[:, 1]) # sparse matrix, n_i_anchor * n_item (old_i_emb_val)
                    feed_dict[model.old_u_cluster] = u_cluster
                    feed_dict[model.old_i_cluster] = i_cluster


                _, bpr_loss, contrastive_loss, l2_reg, dist_loss, mse_user_reg, mse_item_reg  = sess.run([model.ptmzr,
                                                                                                          model.bpr_loss,
                                                                                                          model.contrastive_loss,
                                                                                                          model.reg_loss,
                                                                                                          model.dist_loss,
                                                                                                          model.mse_user_reg,
                                                                                                          model.mse_item_reg],
                                                                                                          feed_dict=feed_dict)
                print('Epoch ', '%04d' % _epoch, 'iter ', '%02d' % iter,
                      'bpr_loss=', '{:.5f}, reg_loss= {:.5f}, dist_loss= {:.5f},'.format(bpr_loss, l2_reg, dist_loss),
                      'contrastive_loss= {:.5f}, cost {:.4f} seconds'.format(contrastive_loss, time.time() - iter_start))
                iter_time.append(time.time() - iter_start)
                logger.write(f"bpr: {np.round(bpr_loss, 5)}, reg: {np.round(l2_reg, 5)}, dist: {np.round(dist_loss, 5)}, con: {np.round(contrastive_loss, 5)}" + '\n')
            
            time_info.append(('finish epoch ' + str(_epoch) + 'training', time.time()))
            time_info_training = sum(iter_time)       
        time_info.append(('finish final epoch training', time.time()))
        time_info.append(('total training time', time_info_training))
        time_info.append(('total eval time', time_info_eval))
        time_info.append(('total sampling time', time_info_sampling))

    sampler.close()
    if parser.log_name:
        logger.write("training time: " + str(sum(iter_time)) + '\n')
        logger.write('best_valid_epoch, best_valid_recall20, best_valid_ndcg20, best_test_recall20, best_test_ndcg20' + '\n')
        logger.write(str([best_valid_epoch, best_valid_recall20, best_valid_ndcg20, best_test_recall20, best_test_ndcg20]) + '\n')
    else:
        print("training time: " + str(sum(iter_time)) + '\n')
        print('best_valid_epoch, best_valid_recall20, best_valid_ndcg20, best_test_recall20, best_test_ndcg20')
        print(str([best_valid_epoch, best_valid_recall20, best_valid_ndcg20, best_test_recall20, best_test_ndcg20]))


def test_model(parser, segment, train_info, val_info, test_info, old_train_set, old_train_matrix, \
                n_epoch, n_old_user=0, n_old_item=0, node_deg_delta=None, logger=None, load_checkpoint='',
                graph_path=None):

    base_model = parser.base_model
    train_set, n_user_train, n_item_train, train_matrix = train_info
    val_set, n_user_val, n_item_val, val_matrix = val_info
    test_set, n_user_test, n_item_test, test_matrix = test_info
    
    if base_model == 'MGCCF':
        user_self_neighs, _ = load_self_neighbours(graph_path[0], 'train', n_user_train, parser.num_neigh, train_matrix)
        item_self_neighs, _ = load_self_neighbours(graph_path[1], 'train', n_item_train, parser.num_neigh, train_matrix.transpose())
        user_self_neighs_val, _ = load_self_neighbours(graph_path[0], 'val', n_user_val, parser.num_neigh, val_matrix)
        item_self_neighs_val, _ = load_self_neighbours(graph_path[1], 'val', n_item_val, parser.num_neigh, val_matrix.transpose())
        user_self_neighs_test, _ = load_self_neighbours(graph_path[0], 'test', n_user_test, parser.num_neigh, test_matrix)
        item_self_neighs_test, _ = load_self_neighbours(graph_path[1], 'test', n_item_test, parser.num_neigh, test_matrix.transpose())

    #prepare train data
    u_adj_dict_train, i_adj_dict_train = sparse_adj_matrix_to_dicts(train_matrix)
    u_adj_list_train, i_adj_list_train = pad_adj(u_adj_dict_train, parser.max_degree, n_item_train), pad_adj(i_adj_dict_train, parser.max_degree, n_user_train)

    u_adj_dict_val, i_adj_dict_val = sparse_adj_matrix_to_dicts(val_matrix)
    u_adj_list_val, i_adj_list_val = pad_adj(u_adj_dict_val, parser.max_degree, n_item_val), pad_adj(i_adj_dict_val, parser.max_degree, n_user_val)

    u_adj_dict_test, i_adj_dict_test = sparse_adj_matrix_to_dicts(test_matrix)
    u_adj_list_test, i_adj_list_test = pad_adj(u_adj_dict_test, parser.max_degree, n_item_test), pad_adj(i_adj_dict_test, parser.max_degree, n_user_test)
    
    print('base model', base_model)
    if base_model == 'MGCCF':
        model = MGCCF(load_checkpoint,
                    [eval(parser.embedded_dimension)[0], n_user_train, n_item_train],
                    eval(parser.embedded_dimension)[1:],
                    parser.max_degree,
                    eval(parser.gcn_sample),
                    ['adam', parser.learning_rate, parser.epsilon],
                    'my_mean',
                    parser.activation,
                    parser.neighbor_dropout,
                    parser.l2,
                    parser.dist_embed,
                    parser.num_self_neigh,
                    parser.num_neg,
                    parser.ui_con_positive,
                    eval(parser.con_ratios),
                    inc_reg = [parser.lambda_mse, parser.lambda_distillation, parser.lambda_global_distill],
                    old_num_user = n_old_user,
                    old_num_item = n_old_item,
                    distill_mode = parser.distill_mode,
                    k_centroids=eval(parser.k_centroids),
                    tau = parser.tau,
                    num_neigh = parser.num_neigh,
                    local_distill_mode = parser.local_mode,
                    contrastive_mode=parser.contrastive_mode,
                    layer_wise=parser.layer_wise,
                    layer_l2_mode=parser.layer_l2_mode,
                    lambda_layer_l2=parser.lambda_layer_l2,
                    lambda_contrastive=eval(parser.lambda_contrastive))

    elif  base_model == 'NGCF':
        model = NGCF(load_checkpoint,
                    [eval(parser.embedded_dimension)[0], n_user_train, n_item_train],
                    eval(parser.embedded_dimension)[1:],
                    parser.max_degree,
                    eval(parser.gcn_sample),
                    ['adam', parser.learning_rate, parser.epsilon],
                    'my_mean',
                    parser.activation,
                    parser.neighbor_dropout,
                    parser.l2,
                    parser.num_neg,
                    parser.ui_con_positive,
                    eval(parser.con_ratios),
                    inc_reg = [parser.lambda_mse, parser.lambda_distillation, parser.lambda_global_distill],
                    old_num_user = n_old_user,
                    old_num_item = n_old_item,
                    distill_mode = parser.distill_mode,
                    k_centroids=eval(parser.k_centroids),
                    tau = parser.tau,
                    local_distill_mode = parser.local_mode,
                    contrastive_mode=parser.contrastive_mode,
                    layer_wise=parser.layer_wise,
                    layer_l2_mode=parser.layer_l2_mode,
                    lambda_layer_l2=parser.lambda_layer_l2,
                    lambda_contrastive=eval(parser.lambda_contrastive))
  
    elif  base_model == 'LightGCN':
        model = LightGCN(load_checkpoint,
                    [eval(parser.embedded_dimension)[0], n_user_train, n_item_train],
                    eval(parser.embedded_dimension)[1:],
                    parser.max_degree,
                    eval(parser.gcn_sample),
                    ['adam', parser.learning_rate, parser.epsilon],
                    parser.batch_pairs,
                    parser.neighbor_dropout,
                    parser.l2,
                    parser.num_neg,
                    parser.ui_con_positive,
                    eval(parser.con_ratios),
                    inc_reg = [parser.lambda_mse, parser.lambda_distillation, parser.lambda_global_distill],
                    old_num_user = n_old_user,
                    old_num_item = n_old_item,
                    distill_mode = parser.distill_mode,
                    k_centroids=eval(parser.k_centroids),
                    tau = parser.tau,
                    local_distill_mode = parser.local_mode,
                    contrastive_mode=parser.contrastive_mode,
                    layer_wise=parser.layer_wise,
                    layer_l2_mode=parser.layer_l2_mode,
                    lambda_layer_l2=parser.lambda_layer_l2,
                    lambda_contrastive=eval(parser.lambda_contrastive))
 
    else:
        NotImplementedError

    sess_config = tf.compat.v1.ConfigProto()

    #sess_config.gpu_options.allow_growth = True 

    with tf.compat.v1.Session(graph=model.graph, config=sess_config) as sess:
        #initialize variables
        sess.run(tf.global_variables_initializer())
        #load checkpoints
        u_emb_val = sess.run(model.user_embeddings)
        i_emb_val = sess.run(model.item_embeddings)

        #loaad existing model
        if load_checkpoint != "":
            #only load existing nodes'embedding
            old_u_emb_val = tf.train.load_variable(load_checkpoint, 'model/user_embedding')
            old_i_emb_val = tf.train.load_variable(load_checkpoint, 'model/item_embedding')

            #If not first segment, u_emb_val would be set as old_embedding + average of similar old user embedding
            #for new users, so for items.
            u_emb_val[:old_u_emb_val.shape[0], ] = old_u_emb_val
            i_emb_val[:old_i_emb_val.shape[0], ] = old_i_emb_val


            # initialize new node as mean of 2-hop neighbors
            if parser.new_node_init == '2hop_mean':
                if n_user_train > n_old_user:
                    new_users_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('user', old_u_emb_val, train_matrix)
                    u_emb_val[n_old_user:,] = new_users_init
                if n_item_train > n_old_item:
                    new_items_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('item', old_i_emb_val, train_matrix)
                    i_emb_val[n_old_item:, ] = new_items_init

            model.user_embeddings.load(u_emb_val, sess)
            model.item_embeddings.load(i_emb_val, sess)                

        best_test_recall20, best_test_ndcg20 = 0., 0.

        if base_model == 'MGCCF':
            precision, t_recall, MAP, t_ndcg, _, _ = evaluate_model(sess, base_model, model, test_info, train_matrix,
                                                                u_adj_list_test, i_adj_list_test, user_self_neighs_test,
                                                                item_self_neighs_test, n_batch_users=parser.batch_evaluate)
        elif base_model == 'NGCF':
            precision, t_recall, MAP, t_ndcg, _, _ = evaluate_model(sess, base_model, model, test_info, train_matrix,
                                                                u_adj_list_test, i_adj_list_test, n_batch_users=parser.batch_evaluate)  
        elif base_model == 'LightGCN':
            precision, t_recall, MAP, t_ndcg, _, _ = evaluate_model(sess, base_model, model, test_info, train_matrix,
                                                                u_adj_list_test, i_adj_list_test, n_batch_users=parser.batch_evaluate)
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


    
def evaluate_model(sess, base_model, model, test_info, train_matrix, u_adj_list, i_adj_list, user_self_neighs=None, item_self_neighs=None, n_batch_users=1024):
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess_config = tf.compat.v1.ConfigProto()
    #sess_config.gpu_options.allow_growth = True
    test_set, n_user, n_item, test_matrix = test_info #n_user: max user index in the test data
    num_batches = int(n_user / n_batch_users) + 1
    user_indexes = np.arange(n_user)
    topk = 100
    precision, recall, MAP, ndcg = [], [], [], []
    pre_list = None
    items = np.arange(0, n_item, 1, dtype=int)

    for batchID in range(num_batches):
        start = batchID * n_batch_users
        end = start + n_batch_users

        if batchID == num_batches - 1:
            if start < n_user:
                end = n_user
            else:
                break
        
        batch_user_index = user_indexes[start:end]
        feed_dict = {}
        feed_dict[model.u_adj_info_ph] = u_adj_list
        feed_dict[model.v_adj_info_ph] = i_adj_list
        if base_model == 'MGCCF':
            feed_dict[model.u_u_graph_ph] = user_self_neighs
            feed_dict[model.v_v_graph_ph] = item_self_neighs
        
        n_user_train, n_item_train = model.num_user, model.num_item
        if n_user > n_user_train: # new user appears in the test set
            old_user_embedding = sess.run(model.user_embeddings)
            new_users_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('user', old_user_embedding, test_matrix)
        else:
            new_users_init = None
        if n_item > n_item_train:
            old_item_embedding = sess.run(model.item_embeddings)
            new_items_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('item', old_item_embedding, test_matrix)
        else:
            new_items_init = None

        rating_preds, user_rep, item_rep = model.predict(batch_user_index, items, n_user, new_users_init, new_items_init)
        rating_preds, user_rep, item_rep = sess.run([rating_preds, user_rep, item_rep], feed_dict)
        train_matrix = train_matrix[:n_user_train, :n_item_train]
        rating_preds[train_matrix[batch_user_index[0]:min(train_matrix.shape[0], batch_user_index[-1]+1)].nonzero()] = 0

        index = np.argpartition(rating_preds, -topk)
        index = index[:, -topk:]
        arr_ind = rating_preds[np.arange(len(rating_preds))[:, None], index]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_preds)), ::-1]
        pred_items = index[np.arange(len(rating_preds))[:, None], arr_ind_argsort]


        if batchID == 0:
            pred_list = pred_items.copy()
        else:
            pred_list = np.append(pred_list, pred_items, axis=0)


    for k in [5, 10, 15, 20]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(batch_ndcg_at_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg, user_rep, item_rep

if __name__  == '__main__':
    time_info = []
    time_info.append(('Program begins', time.time()))

    #parse arguments
    parser = config_baselines.parse_arguments()
    #print('using GPU' + str(parser.device))
    #os.environ['CUDA_VISIBLE_DEVICES'] = parser.device
    parser.load_save_path_prefix = parser.load_save_path_prefix + parser.base_model + '/'
    LOG_SAVE_PATH_PREFIX = parser.load_save_path_prefix
  
    #set seed
    np.random.seed(parser.seed)
    tf.random.set_random_seed(parser.seed)

    save_ratio = ''.join([str(s) for s in eval(parser.con_ratios)])
    save_lambda_con = '-'.join(str(s) for s in eval(parser.lambda_contrastive))
    save_layer_dim = ''.join(str(s) for s in eval(parser.embedded_dimension))
    save_k = ''.join(str(s) for s in eval(parser.k_centroids))
    save_setting = parser.algorithm + f"cp{parser.ui_con_positive}cr{save_ratio}lc{save_lambda_con}" + \
                   f"K{save_k}lr{parser.learning_rate}" + f"bs{parser.batch_pairs}ld{save_layer_dim}" + (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%b_%d_%H_%M_%S")

    #checkpoint and embedding save path
    save_ckpt = LOG_SAVE_PATH_PREFIX + parser.log_folder + '/' + parser.save_cp + '.ckpt' if parser.save_cp else ''
    load_ckpt = LOG_SAVE_PATH_PREFIX + parser.log_folder + '/' + parser.load_cp + '.ckpt' if parser.load_cp else ''

    #loading data
    stream_data_generator = Stream_Data(dataset=parser.dataset, first_segment_time = parser.first_segment_time, last_segment_time = parser.last_segment_time, shuffle=False, test_ratio = parser.test_ratio, \
                                 valid_test_ratio=parser.valid_test_ratio, seed=parser.seed, replay_ratio=parser.replay_ratio, sliding_ratio=parser.sliding_ratio)
    data_segments = stream_data_generator.segments
    time_info.append(('Data loader done', time.time()))
    n_segments = parser.last_segment_time - parser.first_segment_time + 1
    cnt_segment = 0

    #train model
    saved_ckpt = []
    for segment in range(n_segments):
        if segment == 0:
        ## train for the very first segment
            time_info.append(('start first segment training', time.time()))
            if parser.log_name:
                now = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%b_%d_%H_%M_%S") + '-segment' + str(segment)
                name = parser.log_folder + '/' + parser.log_name + '-' + parser.base_model + '-' + parser.dataset
                log_save_path = LOG_SAVE_PATH_PREFIX + name + '/' + now
                result_log_name = parser.log_files + '/' +  parser.base_model + '/' + parser.dataset + '/' + save_setting
                logger = Logger(result_log_name, name, now, parser.load_save_path_prefix)
                logger.open(result_log_name + f'/log.train_segment_{segment}.txt', mode='a')
                for arg in vars(parser):
                    logger.write(arg + '=' + str(getattr(parser, arg)) + '\n')

            else:
                logger = None

            save_ckpt = log_save_path + '/model.ckpt'
            saved_ckpt.append(save_ckpt)

            train_n_user, train_n_item = data_segments[segment]['n_user_train'], data_segments[segment]['n_item_train']
            val_n_user, val_n_item = data_segments[segment]['n_user_val'], data_segments[segment]['n_item_val']
            test_n_user, test_n_item = data_segments[segment]['n_user_test'], data_segments[segment]['n_item_test']

            train_set = data_segments[segment]['train']
            val_set = data_segments[segment]['val']
            test_set = data_segments[segment]['test']

            train_matrix = data_segments[segment]['train_matrix']
            val_matrix = data_segments[segment]['val_matrix']
            test_matrix = data_segments[segment]['test_matrix']

            graph_path = [parser.graph_path + parser.dataset + '/' + 'uu_graph_0.npy', \
                          parser.graph_path + parser.dataset + '/' + 'ii_graph_0.npy']
            print('save_ckpt', save_ckpt)
            train_model(parser,
                        segment,
                        [train_set, train_n_user, train_n_item, train_matrix],
                        [val_set, val_n_user, val_n_item, val_matrix],
                        [test_set, test_n_user, test_n_item, test_matrix],
                        None,
                        None,
                        parser.num_epoch,
                        logger = logger,
                        load_checkpoint=load_ckpt,
                        save_checkpoint=save_ckpt,
                        graph_path=graph_path)
            
            time_info.append(('finish calc mse coef', time.time()))
            load_ckpt = save_ckpt

            cnt_segment += 1

        else:
            time_info.append(('number of segment' + str(segment) + 'begins', time.time()))

            #create logger before training
            if parser.log_name:
                now = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%b_%d_%H_%M_%S") + 'segment' + str(segment)
                name = parser.log_folder + '/' + parser.log_name + '-' + parser.base_model + '-' + parser.dataset 
                log_save_path = LOG_SAVE_PATH_PREFIX + name + '/' + now
                result_log_name = parser.log_files + '/' + parser.base_model + '/' + parser.dataset + '/' + save_setting
                logger = Logger(result_log_name, name, now, parser.load_save_path_prefix)
                logger.open(result_log_name + f'/log.train_segment_{segment}.txt', mode='a')
                for arg in vars(parser):
                    logger.write(arg + '=' + str(getattr(parser, arg)) + '\n')
            else:
                logger = None
            
        ## train for the rest segments
            #assert parser.save_cp == ''
            save_ckpt = log_save_path + '/model.ckpt'
            saved_ckpt.append(save_ckpt)

            next_segment = segment + 1
            prev_segment = segment - 1

            # determine train_set and test_set
            if parser.train_mode == 'acc':
                train_set = data_segments[segment]['acc_train']
            else:
                train_set = data_segments[segment]['train']

            n_user_train, n_item_train = data_segments[segment]['n_user_train'], data_segments[segment]['n_item_train']
            n_old_user_train, n_old_item_train = data_segments[prev_segment]['n_user_train'], data_segments[prev_segment]['n_item_train']
            cur_train_matrix = data_segments[segment]['train_matrix']
            prev_train_set = data_segments[prev_segment]['train']
            prev_train_matrix = data_segments[prev_segment]['train_matrix']

            val_set = data_segments[segment]['val']
            n_item_val, n_user_val = data_segments[segment]['n_item_val'], len(val_set)
            cur_val_matrix = data_segments[segment]['val_matrix']
            test_set = data_segments[segment]['test']
            n_item_test, n_user_test = data_segments[segment]['n_item_test'], len(test_set)
            cur_test_matrix = data_segments[segment]['test_matrix']
            all_set = data_segments[segment]['all']
            n_item_all, n_user_all = data_segments[segment]['n_item_all'], len(all_set)
            cur_all_matrix = data_segments[segment]['all_matrix']


            full_batch_append = ''
            if parser.full_batch:
                assert parser.lambda_mse == 0
                assert parser.reservoir_mode == ''
                assert parser.inc_agg == 0
                assert parser.train_mode == 'acc'
                full_batch_append = '_fb'

            node_deg_delta = None
            
            if parser.base_model == 'MGCCF':
                graph_path = [
                    parser.graph_path + parser.dataset + '/' + 'uu_graph_' + str(segment) + full_batch_append + '.pkl', \
                    parser.graph_path + parser.dataset + '/' + 'ii_graph_' + str(segment) + full_batch_append + '.pkl']
                prev_graph_path = [
                    parser.graph_path + parser.dataset + '/' + 'uu_graph_' + str(prev_segment) + full_batch_append + '.pkl', \
                    parser.graph_path + parser.dataset + '/' + 'ii_graph_' + str(prev_segment) + full_batch_append + '.pkl']

            #calculating regularizer coefficient
            if parser.lambda_mse > 0:
                time_info.append(('start calc mse coef', time.time()))
                u_deg, i_deg = np.array(prev_train_matrix.sum(axis=1)).flatten(), np.array(prev_train_matrix.sum(axis=0)).flatten()
                new_u_deg, new_i_deg = np.array(cur_train_matrix.sum(axis=1)).flatten(), np.array(cur_train_matrix.sum(axis=0)).flatten()
                new_u_deg, new_i_deg = new_u_deg[:n_old_user_train], new_i_deg[:n_old_item_train]
                u_deg_delta, i_deg_delta = u_deg / (new_u_deg + 1e-8), i_deg / (new_i_deg + 1e-8)
                u_deg_norm, i_deg_norm = u_deg_delta / np.linalg.norm(u_deg_delta), i_deg_delta / np.linalg.norm(i_deg_delta)
                delta_n_user = n_user_train - n_old_user_train
                delta_n_item = n_item_train - n_old_item_train
                u_mse_delta = np.concatenate([u_deg_norm, np.zeros(delta_n_user)])
                i_mse_delta = np.concatenate([i_deg_norm, np.zeros(delta_n_item)])
                node_deg_delta = [u_mse_delta, i_mse_delta]

                time_info.append(('finish calc mse coef', time.time()))

            if parser.reservoir_mode != '':
                time_info.append(('start calc reservoir', time.time()))
                assert parser.reservoir_selection != ''
                if parser.reservoir_mode == 'reservoir_sampling':
                    assert parser.sliding_ratio > 0

                print('================', segment, '=================')

                if segment == 1:
                    replay_size = int(stream_data_generator.data_size * parser.replay_ratio)
                    reservoir_size = int(stream_data_generator.data_size * parser.sliding_ratio) if parser.reservoir_mode == 'reservoir_sampling' else None
                    reservoir = Reservoir(data_segments[prev_segment], parser.reservoir_mode, replay_size,
                                          sample_mode=parser.reservoir_selection, merge_mode=parser.union_mode,
                                          sample_per_user=0, reservoir_size=reservoir_size)

                reservoir.set_logger(logger)
                if parser.union_mode == 'snu':
                    assert parser.replay_ratio > 0
                if parser.union_mode == 'uns':
                    union_lists = union_lists_of_list(reservoir.reservoir, train_set)

                elif parser.reservoir_selection in ['uniform', 'inverse_deg', 'prop_deg', 'adp_inverse_deg']:
                    if parser.union_mode == 'snu':
                        if parser.reservoir_selection == 'adp_inverse_deg':
                            new_data_mat = generate_sparse_adj_matrix(train_set, n_user_train, n_item_train)
                        else:
                            new_data_mat = None
                        merged_train_data = reservoir.get_inc_train_data(train_set, new_data_mat=new_data_mat)
                    elif parser.union_mode == 'uns':
                        merged_train_data = reservoir.get_inc_train_data(union_lists, n_new_user=n_user_train, n_new_item=n_item_train, cur_block_train_size=get_list_of_lists_size(train_set))
                    else:
                        raise NotImplementedError

                elif parser.reservoir_selection == 'item_embedding':
                    if parser.union_mode == 'snu':
                        old_i_embedding = tf.train.load_variable(load_ckpt, 'model/item_embedding')[:data_segments[prev_segment]['n_item_train']]
                        #This code block will not be activated
                        merged_train_data = reservoir.get_inc_train_data_item_embedding(train_set, old_i_embedding, cur_train_matrix, 'TODO: manually put a threshold here.')
                    else:
                        raise NotImplementedError
                
                elif parser.reservoir_selection == 'latest':
                    assert parser.union_mode == 'snu'
                    merged_train_data = union_lists_of_list(data_segments[segment]['latest_reservoir'], train_set)

                else:
                    raise NotImplementedError

                if segment < n_segments -1 :
                    print("update reservoir")
                    if parser.reservoir_mode == 'sliding':
                        reservoir.update(all_set, n_user_all, n_item_all,
                                            data_segments[next_segment]['sliding_lists'],
                                            data_segments[next_segment]['sliding_matrix'])
                    else:
                        reservoir.update(all_set, n_user_all, n_item_all)

                train_set = merged_train_data

                time_info.append(('finish calc reservoir', time.time()))

                if parser.inc_agg == 1:
                    assert parser.reservoir_mode == 'sliding'
                    acc_train_list = union_lists_of_list(data_segments[segment]['sliding_lists'], train_set)
                    cur_train_matrix = generate_sparse_adj_matrix(acc_train_list, n_user_train, n_item_train)

            #training  model
            time_info.append(('before enter train model', time.time()))   
            
            #if segment < 7:
            train_model(parser,
                        segment,
                        [train_set, n_user_train, n_item_train, cur_train_matrix],
                        [val_set, n_user_val, n_item_val, cur_val_matrix],
                        [test_set, n_user_test, n_item_test, cur_test_matrix],   
                        prev_train_set,
                        prev_train_matrix,
                        parser.num_epoch,
                        n_old_user=n_old_user_train,
                        n_old_item=n_old_item_train,
                        node_deg_delta = node_deg_delta,
                        logger=logger,
                        load_checkpoint=load_ckpt,
                        save_checkpoint=save_ckpt,
                        graph_path=graph_path)

            load_ckpt = save_ckpt
            """
            else:
                test_model(parser,
                            segment,
                            [train_set, n_user_train, n_item_train, cur_train_matrix],
                            [val_set, n_user_val, n_item_val, cur_val_matrix],
                            [test_set, n_user_test, n_item_test, cur_test_matrix],   
                            prev_train_set,
                            prev_train_matrix,
                            parser.num_epoch,
                            n_old_user=n_old_user_train,
                            n_old_item=n_old_item_train,
                            node_deg_delta = node_deg_delta,
                            logger=logger,
                            load_checkpoint=load_ckpt,
                            graph_path=graph_path)
            """

            cnt_segment += 1
    
    for ckpt in saved_ckpt:
        os.system('rm ' + ckpt + '.*')

    



            




