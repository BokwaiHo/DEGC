from scipy.special import softmax
from scipy.stats import wasserstein_distance
import random
import numpy as np
from data_utils.utils import *
from scipy.sparse import csr_matrix

def lists_to_pairs(lists):
    pair_list = []
    for u in range(len(lists)):
        pair_list += [(u, i) for i in lists[u]]
    random.shuffle(pair_list)
    return pair_list


def weighted_sampling_from_sparse_matrix(input_matrix, output_size, weight=None, per_user=0):
    n_input = len(input_matrix.nonzero()[0])
    #assert output_size <= n_input

    if weight is None:
        weight = input_matrix / input_matrix.sum()

    if per_user == 0:
        rng = np.random.default_rng()
        sampled_idx = rng.choice(np.arange(n_input), size=output_size, replace=True, p=weight.data)  # uniform sample case
        # without replacement
        # sample_matrix = np.zeros(input_matrix.shape).reshape(-1)
        # sample_matrix[sampled_idx] = 1
        # sample_matrix = sample_matrix.reshape(input_matrix.shape[0], input_matrix.shape[1])

        rows = np.array(input_matrix.nonzero()[0][sampled_idx])
        cols = np.array(input_matrix.nonzero()[1][sampled_idx])
        data = np.ones(output_size)
        sample_matrix = csr_matrix((data, (rows, cols)), shape=(input_matrix.shape[0], input_matrix.shape[1]))
        # assert (input_matrix-sample_matrix).data.sum() + output_size == input_matrix.data.sum()
    else:
        # TODO: handle
        pass
        # per_user_size = per_user
        # weight = weight / (weight.sum(axis=1).reshape(-1, 1))
        # sample_matrix = np.zeros(input_matrix.shape)
        # col_idx = np.arange(input_matrix.shape[1])
        # for r in range(input_matrix.shape[0]):
        #     # sampled_idx = np.random.choice(col_idx, size=per_user_size, replace=False, p=weight[r])
        #     sample_matrix[r][sampled_idx] = 1

    return sample_matrix

def weighted_sample_from_lists(input_lists, output_size, n_user, n_item, weight=None):
    # input_matrix = np.array(generate_sparse_adj_matrix(input_lists, n_user, n_item).todense())
    input_matrix = generate_sparse_adj_matrix(input_lists, n_user, n_item)
    sample_matrix = weighted_sampling_from_sparse_matrix(input_matrix, output_size, weight=weight)
    sample_lists = convert_adj_matrix_to_lists(sample_matrix)

    return sample_lists


class Reservoir(object):
    def __init__(self, base_block, reservoir_mode, replay_size, sample_mode, merge_mode='snu', sample_per_user=0, logger=None, reservoir_size=None):
        self.reservoir_mode = reservoir_mode
        self.reservoir_size = reservoir_size # reservoir_size is used when reservoir_mode=reservoir_sampling
        self.replay_size = replay_size # replay_size is the amount of old data to mix with new data
        self.sample_per_user = sample_per_user

        self.sample_mode = sample_mode
        self.merge_mode = merge_mode

        self.logger = logger

        self.reservoir, self.reservoir_matrix, self.n_reservoir_user, self.n_reservoir_item, self.full_data, self.full_data_matrix = self.create_first_reservoir(base_block)
        self.acc_data_size = get_list_of_lists_size(base_block['train'])

    def set_logger(self, logger):
        self.logger = logger

    def log(self, x):
        p = ''
        for s in x:
            p += str(s)
            p += ' '
        if self.logger is None:
            print(p)
        else:
            self.logger.write(p + '\n')

    def create_first_reservoir(self, base_block):
        base_block_user = base_block['n_user_train']
        base_block_item = base_block['n_item_train']
        if self.reservoir_mode == 'full':
            reservoir = base_block['train']
            reservoir_matrix = base_block['train_matrix']
        elif self.reservoir_mode == 'sliding':
            reservoir = base_block['sliding_lists']
            reservoir_matrix = base_block['sliding_matrix']
        else:
            reservoir = weighted_sample_from_lists(base_block['acc_train'], self.reservoir_size, base_block_user, base_block_item, weight=None)
            reservoir_matrix = generate_sparse_adj_matrix(reservoir, base_block_user, base_block_item)

        full_data = base_block['train']
        full_data_matrix = base_block['train_matrix']

        return reservoir, reservoir_matrix, base_block_user, base_block_item, full_data, full_data_matrix


    def get_edge_weight(self, input_matrix):
        input_matrix = input_matrix.astype(np.float64)
        if self.sample_mode == 'uniform':
            weight = input_matrix / input_matrix.sum()

        elif self.sample_mode == 'prop_deg':
            pass

        elif self.sample_mode == 'inverse_deg':
            diag_deg, _ = np.histogram(input_matrix.nonzero()[0], np.arange(input_matrix.shape[0] + 1))
            diag_deg = diag_deg.astype(np.float64)
            mask = diag_deg != 0
            diag_deg = diag_deg.astype(np.float64)
            diag_deg[mask] = 1.0 / diag_deg[mask]
            weight = np.zeros(len(input_matrix.nonzero()[0]))
            source_node_idx = input_matrix.nonzero()[0]
            weight[0] = input_matrix.data[0] * diag_deg[source_node_idx[0]]
            for i in range(input_matrix.data.shape[0]):
                weight[i] = input_matrix.data[i] * diag_deg[source_node_idx[i]]
            weight /= weight.sum()

        else:
            raise NotImplementedError

        return weight

    def get_edge_weight_dense(self, input_matrix, predict_score=None, new_data_mat=None, top_k=0):
        input_matrix = input_matrix.astype(np.float64)

        if self.sample_mode == 'uniform':
            weight = input_matrix / input_matrix.sum()
        elif self.sample_mode == 'prop_deg':
            for r in range(len(input_matrix)):
                input_matrix[r] = input_matrix[r] * input_matrix[r].sum()
            weight = input_matrix / input_matrix.sum()

        elif self.sample_mode == 'inverse_deg':
            for r in range(len(input_matrix)):
                if input_matrix[r].sum() != 0:
                    input_matrix[r] = input_matrix[r] * (1 / input_matrix[r].sum())
            weight = input_matrix / input_matrix.sum()

        elif self.sample_mode == 'adp_inverse_deg':
            new_data_total_edge = new_data_mat.sum()
            old_data_total_edge = input_matrix.sum()
            for r in range(len(input_matrix)):
                if input_matrix[r].sum() != 0:
                    new_data_coef = new_data_mat[r].sum() / new_data_total_edge
                    old_data_coef = input_matrix[r].sum() / old_data_total_edge
                    # adp_coef = max(min(new_data_coef / old_data_coef, 5.0), 0.2) # caps (0.1, 10)
                    adp_coef = new_data_coef / old_data_coef
                    input_matrix[r] = input_matrix[r] * (1 / input_matrix[r].sum()) * adp_coef
            weight = input_matrix / input_matrix.sum()

        elif self.sample_mode == 'boosting_inner_product':
            predict_score = predict_score * input_matrix
            predict_score = -predict_score + np.max(predict_score)
            weight = predict_score / predict_score.sum()

        elif self.sample_mode == 'boosting_wasserstein':
            wasserstein = np.zeros(input_matrix.shape)
            for u in range(len(predict_score)):
                u_hat = softmax(predict_score[u])
                u_approx = softmax(input_matrix[u]) # not real ground truth for u
                                                    # because reservoir is only a
                                                    # subset of all data. this
                                                    # results in false negatives
                wasserstein[u] = wasserstein_distance(u_hat, u_approx)
            weight = wasserstein * input_matrix
            # weight = weight ** 1.2 # sharpen the distribution
            weight = weight / weight.sum()

        elif self.sample_mode == 'mse_distill_score':
            masked_predict_score = input_matrix * predict_score
            # # top-k
            # if the input predict_score is a complete score matrix, do top-k selection
            # if the input predict_score is a row, do weighted sampling according to weights
            if predict_score.shape[0] != 1:
                score_shape = masked_predict_score.shape
                flatten_score = masked_predict_score.reshape(-1)
                flatten_argsort = flatten_score.argsort()[:-top_k]
                flatten_score[flatten_argsort] = 0
                flatten_score[flatten_score.nonzero()] = 1
                masked_predict_score = flatten_score.reshape(score_shape)
            weight = masked_predict_score / masked_predict_score.sum()

        else:
            raise NotImplementedError

        return weight

    def sample_and_union(self, new_data_lists, predict_score, new_data_mat=None):
        # sample part
        # weight = self.get_edge_weight_dense(self.reservoir_matrix, predict_score, new_data_mat=new_data_mat, top_k=self.replay_size)
        weight = self.get_edge_weight(self.reservoir_matrix)

        self.log(['............printing weights.........'])
        self.log(["sample size:", self.replay_size])
        self.log(['mode:', self.sample_mode])

        weight_nonzero = weight[weight.nonzero()]
        self.log(['max, min, mean, std:', weight_nonzero.max(), weight_nonzero.min(), weight_nonzero.mean(), weight_nonzero.std()])
        
        # sample_reservoir_mat = weighted_sampling_from_dense_matrix(reservoir_matrix_dense, self.replay_size, weight=weight, per_user=self.sample_per_user)

        sample_reservoir_mat = weighted_sampling_from_sparse_matrix(self.reservoir_matrix, self.replay_size, weight=weight, per_user=self.sample_per_user)

        self.log(["sum of sampled pairs:", sample_reservoir_mat.sum()])
        self.log(['....................................'])      

        # sample without replacement
        sampled_reservoir_list = convert_adj_matrix_to_lists(sample_reservoir_mat)
   
        # union part
        result_lists = union_lists_of_list(sampled_reservoir_list, new_data_lists)

        return result_lists#, sample_reservoir_mat

    def update(self, new_data_lists, n_new_user, n_new_item, pre_computed_reservoir_lists=None, pre_computed_reservoir_matrix=None):

        self.full_data = union_lists_of_list(self.full_data, new_data_lists)
        self.full_data_matrix = generate_sparse_adj_matrix(self.full_data, n_new_user, n_new_item)
        if self.reservoir_mode == 'full':
            self.reservoir = self.full_data
            self.reservoir_matrix = self.full_data_matrix
            self.n_reservoir_user = n_new_user
            self.n_reservoir_item = n_new_item
        elif self.reservoir_mode == 'sliding':
            self.reservoir = pre_computed_reservoir_lists
            self.reservoir_matrix = pre_computed_reservoir_matrix
            assert pre_computed_reservoir_matrix.shape[0] == n_new_user
            assert pre_computed_reservoir_matrix.shape[1] == n_new_item
            self.n_reservoir_user = n_new_user
            self.n_reservoir_item = n_new_item
        elif self.reservoir_mode == 'reservoir_sampling':
            # for case there is a fixed sized reservoir - reservoir sampling algo
            # https://en.wikipedia.org/wiki/Reservoir_sampling
            # used in https://arxiv.org/pdf/2007.02747.pdf (potential baseline)
            for i in range(n_new_user-len(self.reservoir)):
                self.reservoir.append([])
            
            new_pair_list = []
            for u in range(len(new_data_lists)):
                new_pair_list += [(u, i) for i in new_data_lists[u]]
            random.shuffle(new_pair_list)

            for i in range(self.acc_data_size, self.acc_data_size+len(new_pair_list)):
                j = np.random.randint(0, i)
                if j < self.reservoir_size:
                    rand_u = np.random.randint(0, len(self.reservoir))
                    while len(self.reservoir[rand_u]) <= 0:
                        rand_u = np.random.randint(0, len(self.reservoir))
                    rand_i = np.random.randint(0, len(self.reservoir[rand_u]))
                    cur_new_pair = new_pair_list[i-self.acc_data_size]
                    self.reservoir[rand_u].pop(rand_i)
                    self.reservoir[cur_new_pair[0]].append(cur_new_pair[1])
            
            self.reservoir_matrix = generate_sparse_adj_matrix(self.reservoir, n_new_user, n_new_item)
            self.acc_data_size += len(new_pair_list)
            self.n_reservoir_user = n_new_user
            self.n_reservoir_item = n_new_item
        else:
            raise NotImplementedError

    def get_inc_train_data(self, new_data_lists, predict_score=None, n_new_user=None, n_new_item=None, new_data_mat=None, cur_block_train_size=0):
        if self.merge_mode == 'snu':
            return self.sample_and_union(new_data_lists, predict_score, new_data_mat=new_data_mat)
        elif self.merge_mode == 'uns':
            assert n_new_user is not None and n_new_item is not None
            assert cur_block_train_size != 0
            # union_matrix_dense = np.array(generate_sparse_adj_matrix(new_data_lists, n_new_user, n_new_item).todense())
            # weight = self.get_edge_weight(union_matrix_dense, predict_score)
            # sample_reservoir_mat = weighted_sampling_from_dense_matrix(union_matrix_dense, self.replay_size+cur_block_train_size, weight=weight, per_user=self.sample_per_user)

            union_matrix_sparse = generate_sparse_adj_matrix(new_data_lists, n_new_user, n_new_item)
            weight = self.get_edge_weight(union_matrix_sparse, predict_score)
            sample_reservoir_mat = weighted_sampling_from_sparse_matrix(union_matrix_sparse,
                                                                       self.replay_size + cur_block_train_size,
                                                                       weight=weight, per_user=self.sample_per_user)
            return convert_adj_matrix_to_lists(sample_reservoir_mat)
        else:
            raise NotImplementedError



'''
    def get_one_hop_mean(self, n_node, node_embedding, train_matrix):
        nodes_interest = np.zeros([n_node, 128])
        n
        train_matrix = train_matrix.toarray()
        for node in range(n_node):
            neighbour_ids = np.where(train_matrix[node] != 0)[0]
            neighbour_ids = [neighbour for neighbour in neighbour_ids if neighbour < node_embedding.shape[0]]
            if len(neighbour_ids) == 0:
                neighbour_ids = list(np.random.choice(list(range(node_embedding.shape[1])), 128, replace=False))
            nodes_interest[node] = np.mean(np.take(node_embedding, neighbour_ids, axis=0), axis=0)
        return nodes_interest

    def get_adaptive_reservoir_size(self, new_train_lists, old_item_embedding, new_train_matrix, mode='degree', old_user_rep=None, old_item_rep=None, new_user_rep=None, new_item_rep=None, old_user_embedding=None):
        old_n_user, old_n_item = self.reservoir_matrix.shape[0], self.reservoir_matrix.shape[1]
        print(old_n_item, old_item_embedding.shape)
        assert old_n_item == old_item_embedding.shape[0]

        if mode == 'euc' or mode == 'degree':
            new_item_initial_embedding = get_node_init_embedding_by_aggregating_two_hop_neighbours('item', old_item_embedding, new_train_matrix)
            item_embedding = np.concatenate((old_item_embedding, new_item_initial_embedding), axis=0)
            
            # normalization
            item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
            old_item_embedding = old_item_embedding / np.linalg.norm(old_item_embedding, axis=1, keepdims=True)
            
            users_new_interest, active_user_set = self.get_user_interest(old_n_user, item_embedding, train_lists=new_train_lists)
            users_old_interest, _ = self.get_user_interest(old_n_user, old_item_embedding, self.reservoir_matrix)
            
            active_user_list = list(active_user_set)
            users_old_interest, users_new_interest = users_old_interest[active_user_list], users_new_interest[active_user_list]
        
        if mode == 'euc':
            users_interest_diff = np.linalg.norm(users_new_interest-users_old_interest, axis=1)
            # users_old_interest_length = np.linalg.norm(users_old_interest, axis=1)
            # reservoir_ratio = np.mean(users_interest_diff / users_old_interest_length)
            reservoir_ratio = np.mean(users_interest_diff) * 6.3224 #0.1087

            # t = users_interest_diff / users_old_interest_length
            # print(t.min(), t.max())
            # print(t.mean())
            # print(t.std())
            # save_pickle(t, 'temp/', 'euc_ratio')

        if mode == 'degree':
            unit_users_new_interest = users_new_interest / np.expand_dims(np.linalg.norm(users_new_interest, axis=1), 1)
            unit_users_old_interest = users_old_interest / np.expand_dims(np.linalg.norm(users_old_interest, axis=1), 1)
            dot_products = np.einsum('ij,ij->i', unit_users_new_interest, unit_users_old_interest)
            angles = np.arccos(dot_products) 

            # mask = np.zeros(old_n_user)
            # mask[active_user_list] = angles

            # print(angles.min(), angles.max())
            # print(angles.mean())
            # print(angles.std())
            # save_pickle(angles, 'temp/', 'angles')

            # assert False

            reservoir_ratio = np.mean(angles) / np.pi 
            # assert False

        if mode == 'inference':
            new_user_rep = new_user_rep[:old_n_user]
            new_item_rep = new_item_rep[:old_n_item]

            # normalization
            new_user_rep = new_user_rep / np.linalg.norm(new_user_rep, axis=1, keepdims=True)
            new_item_rep = new_item_rep / np.linalg.norm(new_item_rep, axis=1, keepdims=True)
            old_user_rep = old_user_rep / np.linalg.norm(old_user_rep, axis=1, keepdims=True)
            old_item_rep = old_item_rep / np.linalg.norm(old_item_rep, axis=1, keepdims=True)

            user_mse = np.linalg.norm(new_user_rep - old_user_rep, axis=1)
            item_mse = np.linalg.norm(new_item_rep - old_item_rep, axis=1)

            # score = np.concatenate((user_mse, item_mse))
            # reservoir_ratio = np.mean(score) * 3.4964
            # print('score_1')
            # print(np.mean(score_1))
            # print(score_1.min(), score_1.max())
            # print(score_1.std())

            # print('score_2_user')
            reservoir_ratio = (np.mean(user_mse) + np.mean(item_mse)) # * 1.466 # 18.85 #*

        if mode == 'interest_shift':
            print("full size: ", new_train_matrix.shape)


            new_train_lists = union_lists_of_list(self.reservoir, new_train_lists)
            new_train_matrix = generate_sparse_adj_matrix(new_train_lists, new_train_matrix.shape[0], new_train_matrix.shape[1])
            old_train_matrix = self.reservoir_matrix

            new_train_matrix_transposed = new_train_matrix.T
            old_train_matrix_transposed = old_train_matrix.T

            # normalization
            old_item_embedding = old_item_embedding / np.linalg.norm(old_item_embedding, axis=1, keepdims=True)
            old_user_embedding = old_user_embedding / np.linalg.norm(old_user_embedding, axis=1, keepdims=True)
            
            users_new_interest = self.get_one_hop_mean(old_train_matrix.shape[0], old_item_embedding, new_train_matrix)
            users_old_interest = self.get_one_hop_mean(old_train_matrix.shape[0], old_item_embedding, old_train_matrix)

            items_new_interest = self.get_one_hop_mean(old_train_matrix_transposed.shape[0], old_user_embedding, new_train_matrix_transposed)
            items_old_interest = self.get_one_hop_mean(old_train_matrix_transposed.shape[0], old_user_embedding, old_train_matrix_transposed)
        
            users_interest_diff = np.sum(np.linalg.norm(users_new_interest-users_old_interest, axis=1))
            items_interest_diff = np.sum(np.linalg.norm(items_new_interest-items_old_interest, axis=1))

            reservoir_ratio = users_interest_diff + items_interest_diff
            # reservoir_ratio = ((reservoir_ratio - 6278.09111397)/6278.09111397 * 10  + 1) * 3 # Gowalla
            # reservoir_ratio = ((reservoir_ratio - 14362.0630948)/14362.0630948 * 10 + 1) * 0.1 # TB2015
            reservoir_ratio = ((reservoir_ratio - 6057.844804568)/6057.844804568 * 10 + 1) * 0.2 # Yelp

        self.log(['adaptive reservoir ratio: ', reservoir_ratio])

        return reservoir_ratio
        
    def weighted_sampling_from_dense_matrix(input_matrix, output_size, weight=None, per_user=0):
        n_input = np.count_nonzero(input_matrix)
        assert output_size <= n_input

        if weight is None:
            weight = input_matrix / input_matrix.sum()
    
        if per_user == 0:
            indices = np.arange(input_matrix.nonzero()[0].max() * input_matrix.nonzero()[1].max())
            rng = np.random.default_rng()
            sampled_idx = rng.choice(indices, size=output_size, replace=False, p=weight.reshape(-1))

            # with replacement
            # sample_matrix = []
            # for i in range(input_matrix.shape[0]):
            #     sample_matrix.append([])
            # for i in sampled_idx:
            #     user = i // input_matrix.shape[1]
            #     item = i % input_matrix.shape[1]
            #     sample_matrix[user].append(item)

            # without replacement
            sample_matrix = np.zeros(input_matrix.shape).reshape(-1)
            sample_matrix[sampled_idx] = 1
            sample_matrix = sample_matrix.reshape(input_matrix.shape[0], input_matrix.shape[1])
        else:
            per_user_size = per_user
            weight = weight / (weight.sum(axis=1).reshape(-1,1))
            sample_matrix = np.zeros(input_matrix.shape)
            col_idx = np.arange(input_matrix.shape[1])
            for r in range(input_matrix.shape[0]):
                # sampled_idx = np.random.choice(col_idx, size=per_user_size, replace=False, p=weight[r])
                sample_matrix[r][sampled_idx] = 1

        return sample_matrix
        
    def get_inc_train_data_item_embedding(self, new_train_lists, old_i_embedding, new_train_matrix, threshold):
        old_train_matrix = self.reservoir_matrix
        old_n_user, old_n_item = old_train_matrix.shape[0], old_train_matrix.shape[1]
        assert old_n_item == old_i_embedding.shape[0]

        new_item_initial_embedding = get_node_init_embedding_by_aggregating_two_hop_neighbours('item', old_i_embedding, new_train_matrix, seed=self.seed)
        item_embedding = np.concatenate((old_i_embedding, new_item_initial_embedding), axis=0)
        users_new_interest, active_user_set = self.get_user_interest(old_n_user, item_embedding, train_lists=new_train_lists, seed=self.seed)

        # select according to the threshold
        # TODO: fix this, leads to memory leak
        old_train_matrix = old_train_matrix.toarray()
        reservoir = []
        user2item, item2user = sparse_adj_matrix_to_dicts(old_train_matrix)
        for u in range(old_n_user):
            x = np.where(old_train_matrix[u] != 0)[0]
            item_ids = user2item[u]
            print(x)
            print(item_ids)
            exit()
            if len(item_ids) == 0 or u not in active_user_set: 
                reservoir.append([])
                continue
            user_old_interest = np.mean(np.take(old_i_embedding, item_ids, axis=0), axis=0).reshape(1, 128)
            user_new_interest = users_new_interest[u].reshape(1, 128)

            old_items_embedding = np.take(old_i_embedding, item_ids, axis=0)
            old_interest_diff = ((old_items_embedding - user_old_interest)**2).mean(axis=1)
            new_interest_diff = ((old_items_embedding - user_new_interest)**2).mean(axis=1)
            item_weight = new_interest_diff - old_interest_diff
            over_threshold_item = item_ids[np.where(np.where(item_weight > threshold, 1, 0) == 1)]
            reservoir.append(list(over_threshold_item))        

        reservoir_size_ = get_list_of_lists_size(reservoir)
        new_data_size = get_list_of_lists_size(new_train_lists)
        self.log(['item_embedding reservoir size: ', reservoir_size_])
        self.log(['item_embedding reservoir ratio w.r.t. inc. block size: ', reservoir_size_/new_data_size])
        inc_train_data = union_lists_of_list(reservoir, new_train_lists)

        return inc_train_data    

    def get_user_interest(self, n_user, item_embedding, train_matrix=None, train_lists=None, emb_size=128):
        users_interest = np.zeros([n_user, emb_size])
        active_user_set = set()
        if train_lists is not None:
            for user, items in enumerate(train_lists):
                if user >= n_user:
                    break
                # items = [item for item in items if item < item_embedding.shape[0]]
                if len(items) != 0:
                    active_user_set.add(user)
                    users_interest[user] = np.mean(np.take(item_embedding, items, axis=0), axis=0)
        else:
            # TODO: fix this, leads to memory leak
            train_matrix = train_matrix.toarray()
            for u in range(n_user):
                item_ids = np.where(train_matrix[u] != 0)[0]
                if len(item_ids) == 0:
                    rng = np.random.default_rng(seed)
                    item_ids = list(rng.choice(list(range(item_embedding.shape[1])), emb_size, replace=False))
                users_interest[u] = np.mean(np.take(item_embedding, item_ids, axis=0), axis=0)
        return users_interest, active_user_set
'''
