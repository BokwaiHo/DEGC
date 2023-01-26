import numpy as np
import pickle as pkl
import scipy.sparse as sp
import random

from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def save_pickle(obj, path, name):
    with open(path + name + '.pkl', 'wb') as f:
        pkl.dump(obj, f)


def load_pickle(path, name):
    with open(path + name, 'rb') as f:
        return pkl.load(f, encoding='latin1')


def split_data_randomly(user_records, test_ratio, seed=0):
    '''
    Split each user's item neighbors to train set and test set with certain ratio
    :param user_records: list of lists of user's item neighbors
    :param test_ratio: ratio of test
    :param seed: seed
    :return: train_set and test_set after splitting
    '''
    if test_ratio == 0:
        return user_records, [[] for i in range(len(user_records))]

    train_set = []
    test_set = []
    for user_id, item_list in enumerate(user_records):

        tmp_train_sample, tmp_test_sample = train_test_split(item_list, test_size=test_ratio, random_state=seed)
        train_set.append(tmp_train_sample)
        test_set.append(tmp_test_sample)

    return train_set, test_set


def generate_sparse_adj_matrix(train_set, num_users, num_items, user_shift=0, item_shift=0):
    '''
    Generate user_item sparse adjacent matrix
    :param train_set: list of lists of users' test item neighbors (length of number of users)
    :param num_users: number of users
    :param num_items: number of items
    :param user_shift: user index shift number
    :param item_shift: item index shift number
    :return: sprase adjacent matrix
    '''
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, article_list in enumerate(train_set):
        for article in article_list:
            row.append(user_id + user_shift)
            col.append(article + item_shift)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = sp.csr_matrix((data, (row, col)), shape=(num_users, num_items))
    return rating_matrix


def adj_matrix_to_dicts(train_matrix):
    '''
    Create dictionary of user's item neighbor dictionary and item's user neighbor dictionary from adjacent matrix
    :param train_matrix: adjacency matrix U * I
    :return user2item: user as key, item neighbors as index
    :return item2user: item as key, user neighbors as index
    '''
    user2item, item2user = {}, {}
    user_item_ratings = train_matrix.toarray()
    for i in range(len(user_item_ratings)):
        neigh_items = np.where(user_item_ratings[i] != 0)[0].tolist()
        user2item[i] = neigh_items

    item_user_ratings = user_item_ratings.transpose()
    for j in range(len(item_user_ratings)):
        neigh_users = np.where(item_user_ratings[j] != 0)[0].tolist()
        item2user[j] = neigh_users

    return user2item, item2user

def sparse_adj_matrix_to_dicts(train_matrix):
    '''
    Create dictionary of user's item neighbor dictionary and item's user neighbor dictionary from sparse adjacency matrix
    :param train_matrix: sparse adjacency matrix
    :return user2item: user as key, item neighbors as index
    :return item2user: item as key, user neighbors as index
    '''
    user2item, item2user = {}, {}
    user_idx = train_matrix.nonzero()[0]
    item_idx = train_matrix.nonzero()[1]

    for i in range(train_matrix.shape[0]):
        user2item[i] = []
    for i in range(train_matrix.shape[1]):
        item2user[i] = []

    for i, user in enumerate(user_idx):
        item = item_idx[i]
        if user not in user2item:
            user2item[user] = []
        if item not in item2user:
            item2user[item] = []
        user2item[user].append(item)
        item2user[item].append(user)

    return user2item, item2user

def adj_matrix_to_dicts_(train_matrix, n_user, n_item):
    user2item, item2user = {}, {}
    user2noitem, item2nouser = {}, {}
    user_item_ratings = train_matrix.toarray()
    for i in range(len(user_item_ratings)):
        neigh_items = np.where(user_item_ratings[i] != 0)[0].tolist()
        neigh_no_items = list(set(range(n_item)).difference(set(neigh_items)))
        user2item[i] = neigh_items
        user2noitem[i] = neigh_no_items

    item_user_ratings = user_item_ratings.transpose()
    for j in range(len(item_user_ratings)):
        neigh_users = np.where(item_user_ratings[j] != 0)[0].tolist()
        neigh_no_users = list(set(range(n_user)).difference(set(neigh_users)))
        item2user[j] = neigh_users
        item2nouser[j] = neigh_no_users
    return user2item, item2user

def get_empty_keys_from_dict(input_dict):
    '''
    Get list of keys whose values are empty
    :param input_dict: a dictionary
    :return: list of keys
    '''
    empty_keys = []
    for key in input_dict:
        if len(input_dict[key]) == 0:
            empty_keys.append(key)
    return empty_keys
    

def convert_sparse_matrix_to_sparse_tensor_input(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return (indices, coo.data, coo.shape)


def convert_adj_matrix_to_lists(adj_matrix):
    lists = []
    for u, i_list in enumerate(adj_matrix):
        lists.append(list(i_list.nonzero()[0]))
    return lists

def get_list_of_lists_size(lists):
    n = 0
    for l in lists:
        n += len(l)
    return n

def sample_reservoir_by_size(adj_matrix, output_size, input_size=None, p=None, per_user_split=False):
    if p is None:
        p = adj_matrix / adj_matrix.sum()

    n_interaction = np.count_nonzero(adj_matrix)
    assert output_size <= n_interaction
    sample_ratio = output_size / n_interaction
    sample_matrix = np.zeros(adj_matrix.shape)

    if per_user_split:
        for u, i_list in enumerate(adj_matrix):
            n_i_list = np.count_nonzero(i_list)
            if n_i_list > 0:
                n_sample = int(np.round(sample_ratio * n_i_list))
                sample_idx = np.random.choice(np.arange(adj_matrix.shape[1]), size=n_sample, replace=False, p=p[u]/p[u].sum())
                sample_matrix[u][sample_idx] = 1

        n_sample_interaction = np.count_nonzero(sample_matrix)    
        # pad some data to match output_size
        if n_sample_interaction < output_size:
            diff_matrix = (adj_matrix - sample_matrix).reshape(-1)
            diff_matrix_prob = diff_matrix / np.count_nonzero(diff_matrix)
            sample_idx = np.random.choice(np.arange(diff_matrix_prob.shape[0]), size=output_size-n_sample_interaction, replace=False, p=diff_matrix_prob)
            for sample in sample_idx:
                u = sample // adj_matrix.shape[1]
                i = sample % adj_matrix.shape[1]
                sample_matrix[u,i] = 1
        # remove some data to match output_size
        if n_sample_interaction > output_size:
            diff_matrix_prob = sample_matrix.reshape(-1) / n_sample_interaction
            sample_idx = np.random.choice(np.arange(diff_matrix_prob.shape[0]), size=n_sample_interaction-output_size, replace=False, p=diff_matrix_prob)
            for sample in sample_idx:
                u = sample // adj_matrix.shape[1]
                i = sample % adj_matrix.shape[1]
                sample_matrix[u,i] = 0
    else:
        indices = np.arange(adj_matrix.shape[0] * adj_matrix.shape[1])
        sampled_idx = np.random.choice(indices, size=output_size, replace=False, p=p.reshape(-1))
        for idx in sampled_idx:
            u = idx // adj_matrix.shape[1]
            i = idx % adj_matrix.shape[1]
            sample_matrix[u, i] = 1
            
    return sample_matrix

def update_reservoir(reservoir, new_set, n_user, n_item, mode='uniform', reservoir_size=None, per_user_split=False, acc_data_size=0):
    new_reservoir = []
    if reservoir_size is None:
        reservoir_size = 0
        for i in reservoir:
            reservoir_size += len(i)

    if mode == 'union':
        if new_set is not None:
            for u in range(len(new_set)):
                if u < len(reservoir):
                    new_reservoir.append(new_set[u] + reservoir[u])
                else:
                    new_reservoir.append(new_set[u].copy())
        else:
            new_reservoir = reservoir

    elif mode == 'uniform':
        if new_set is not None: # reservoir sampling method is used
            assert acc_data_size != 0

            for i in range(n_user-len(reservoir)):
                reservoir.append([])
           
            pair_list = []
            for u in range(len(new_set)):
                pair_list += [(u, i) for i in new_set[u]]

            random.shuffle(pair_list)

            k = 0
            for i in range(acc_data_size, acc_data_size+len(pair_list)):
                j = np.random.randint(0, i)
                if j < reservoir_size:
                    rand_u = np.random.randint(0, len(reservoir))
                    while len(reservoir[rand_u]) <= 0:
                        rand_u = np.random.randint(0, len(reservoir))
                    rand_i = np.random.randint(0, len(reservoir[rand_u]))
                    reservoir[rand_u].pop(rand_i)
                    reservoir[pair_list[i-acc_data_size][0]].append(pair_list[i-acc_data_size][1])
                    k+=1
            new_reservoir = reservoir

        else:
            new_reservoir = reservoir
            new_reservoir_matrix = np.array(generate_sparse_adj_matrix(new_reservoir, n_user, n_item).todense())
            new_reservoir = convert_adj_matrix_to_lists(sample_reservoir_by_size(new_reservoir_matrix, reservoir_size, per_user_split=per_user_split))

    else:
        raise NotImplementedError

    return new_reservoir

def select_k_per_user(data_set, k, i_emb):
    '''
    Select k nearest items from each user's neighborhood
    :param data_set: list of lists of user's item neighbors
    :param k: number of neighbors selected
    :param i_emb: item embedding
    :return: list of lists of user's selected item neighbors
    '''
    selected_set = []
    for u, i_list in enumerate(data_set):
        if len(i_list) <= k:
            selected_set.append(i_list.copy())
        else:
            i_list_emb = np.take(i_emb, i_list, axis=0)
            kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=10).fit(i_list_emb)
            selected_idx, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, i_list_emb)
            selected_set.append(list(np.array(i_list)[selected_idx]))
    return selected_set

def union_lists_of_list(old_lists, new_lists):
    '''
    Merge two lists of which same index indicates same users/items
    :param old_lists: old list of lists
    :param new_lists:  new list of lists
    :return: union list of lists
    '''
    union_lists = []
    for u in range(len(new_lists)):
        if u < len(old_lists):
            union_lists.append(new_lists[u] + old_lists[u])
        else:
            union_lists.append(new_lists[u].copy())
    return union_lists


def get_two_hop_neighbours_of_new_nodes(node_type, old_n_node, new_train_matrix, emb_size=128):
    assert node_type == 'user' or node_type == 'item'
    if node_type == 'item':
        new_train_matrix = new_train_matrix.transpose()
    new_n_node = new_train_matrix.get_shape()[0]
    new_node_two_hop_neighbours = {}

    user2item, item2user = sparse_adj_matrix_to_dicts(new_train_matrix)

    for new_node in range(old_n_node, new_n_node):
        if new_node not in new_node_two_hop_neighbours:
            new_node_two_hop_neighbours[new_node] = []
        for i in user2item[new_node]:
            new_node_two_hop_neighbours[new_node] += item2user[i]
        new_node_two_hop_neighbours[new_node] = list(set(new_node_two_hop_neighbours[new_node]))
        if len(new_node_two_hop_neighbours[new_node]) == 0:
            new_node_two_hop_neighbours[new_node] = list(np.random.choice(list(range(old_n_node)), emb_size, replace=False))
        new_node_two_hop_neighbours[new_node] = [i for i in new_node_two_hop_neighbours[new_node] if i < old_n_node]

    return new_node_two_hop_neighbours

def get_node_init_embedding_by_aggregating_two_hop_neighbours(node_type, old_node_embedding, new_train_matrix, emb_size=128):
    assert node_type == 'user' or node_type == 'item'
    old_n_node = old_node_embedding.shape[0]
    new_n_node = new_train_matrix.get_shape()[0] if node_type == 'user' else new_train_matrix.get_shape()[1]

    new_node_two_hop_neighbours = get_two_hop_neighbours_of_new_nodes(node_type, old_n_node, new_train_matrix)
    new_node_initial_embedding = np.zeros([new_n_node-old_n_node, emb_size])
    for node in range(old_n_node, new_n_node):
        node_two_hop_embeddings = np.take(old_node_embedding, new_node_two_hop_neighbours[node], axis=0)
        new_node_initial_embedding[node-old_n_node] = np.mean(node_two_hop_embeddings, axis=0)

    return new_node_initial_embedding


# def get_two_hop_neighbours_of_new_nodes(node_type, old_n_node, new_train_matrix):
#     assert node_type == 'user' or node_type == 'item'
#     if node_type == 'item':
#         new_train_matrix = new_train_matrix.transpose()
#     new_n_node = new_train_matrix.shape[0]
#     new_node_two_hop_neighbours = {}
#     new_train_matrix_transposed = new_train_matrix.transpose()
#
#     for new_node in range(old_n_node, new_n_node):
#         one_hop_neighbours = np.where(new_train_matrix[new_node] != 0)[0].tolist()
#         two_hop_neighbours = set()
#         for neighbour in one_hop_neighbours:
#             two_hop_neighbours.update(np.where(new_train_matrix_transposed[neighbour] != 0)[0].tolist())
#         two_hop_neighbours = [i for i in two_hop_neighbours if i < old_n_node]
#         if len(two_hop_neighbours) == 0:
#             two_hop_neighbours = list(np.random.choice(list(range(old_n_node)), 128, replace=False))
#         new_node_two_hop_neighbours[new_node] = two_hop_neighbours
#
#     return new_node_two_hop_neighbours
#
#
# def get_node_init_embedding_by_aggregating_two_hop_neighbours(node_type, old_node_embedding, new_train_matrix):
#     assert node_type == 'user' or node_type == 'item'
#     old_n_node = old_node_embedding.shape[0]
#     new_n_node = new_train_matrix.shape[0] if node_type == 'user' else new_train_matrix.shape[1]
#
#     new_node_two_hop_neighbours = get_two_hop_neighbours_of_new_nodes(node_type, old_n_node, new_train_matrix.toarray())
#     new_node_initial_embedding = np.zeros([new_n_node - old_n_node, 128])
#     for node in range(old_n_node, new_n_node):
#         node_two_hop_embeddings = np.take(old_node_embedding, new_node_two_hop_neighbours[node], axis=0)
#         new_node_initial_embedding[node - old_n_node] = np.mean(node_two_hop_embeddings, axis=0)
#
#     return new_node_initial_embedding

# ====================================================================
# ====================== archived functions =========================
# ======================== not used anymore ==========================
# ====================================================================

# def shrink_by_size(input_set, output_size, n_user, n_item, input_size=None, mode='per-user', p=None):
#     if input_size is None:
#         original_size = 0
#         for i in input_set:
#             original_size += len(i)

#     if output_size < original_size:
#         split_ratio = output_size / original_size

#         if mode == 'per-user':
#             new_set = split_data_randomly(input_set, 1-split_ratio, seed=0)[0]

#             new_matrix = np.array(generate_sparse_adj_matrix(new_set, n_user, n_item).todense())
#             input_matrix = np.array(generate_sparse_adj_matrix(input_set, n_user, n_item).todense())

#             # pad some data
#             if new_matrix.sum() < output_size:
#                 diff_matrix = (input_matrix - new_matrix).reshape(-1)
#                 diff_matrix_prob = diff_matrix / diff_matrix.sum()
#                 sample_idx = np.random.choice(np.arange(diff_matrix_prob.shape[0]), size=output_size-int(new_matrix.sum()), replace=False, p=diff_matrix_prob)
#                 for sample in sample_idx:
#                     u = sample // n_item
#                     i = sample % n_item
#                     new_set[u].append(i)
#             # remove some data
#             if new_matrix.sum() > output_size:
#                 diff_matrix_prob = new_matrix.reshape(-1) / new_matrix.sum()
#                 sample_idx = np.random.choice(np.arange(diff_matrix_prob.shape[0]), size=(int(new_matrix.sum())-output_size), replace=False, p=diff_matrix_prob)
#                 for sample in sample_idx:
#                     u = sample // n_item
#                     i = sample % n_item
#                     new_set[u].remove(i)

#         elif mode == 'random':
#             raise NotImplementedError
#         else:
#             raise NotImplementedError

#     else:
#         new_set = input_set
#     return new_set
