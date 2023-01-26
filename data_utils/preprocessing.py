from copy import deepcopy


def generate_unique_mappings(data):
    '''
    Create user/item_index and index_user/item dictionary
    :param data: dataframe of data with attributes such as user_id, item_id
    :return u_mapping: user as key and index as value
    :return i_mapping: item as key and index as value
    :return inv_u_mapping: user as value and index as key
    :return inv_i_mapping: item as value and index as key
    '''
    unique_users = data['user_id'].unique()
    unique_items = data['item_id'].unique()
    u_mapping = {k: v for v, k in enumerate(unique_users)}
    i_mapping = {k: v for v, k in enumerate(unique_items)}
    inv_u_mapping = {v: k for k, v in u_mapping.items()}
    inv_i_mapping = {v: k for k, v in i_mapping.items()}
    return u_mapping, i_mapping, inv_u_mapping, inv_i_mapping


def update_mappings(data, u_mapping, i_mapping, inv_u_mapping, inv_i_mapping):
    '''
    Add new_users and items to the user/item_index and index_user/item dictionary
    :param data: dataframe of data with attributes such as user_id, item_id
    :param u_mapping: user as key and index as value
    :param i_mapping: item as key and index as value
    :param inv_u_mapping: user as value and index as key
    :param inv_i_mapping: item as value and index param
    :return: updated dictionaries
    '''
    new_users = data['user_id'].unique()
    new_items = data['item_id'].unique()
    u_count, i_count = len(u_mapping), len(i_mapping)
    for u in new_users:
        if u not in u_mapping:
            u_mapping[u] = u_count
            inv_u_mapping[u_count] = u
            u_count += 1
    for i in new_items:
        if i not in i_mapping:
            i_mapping[i] = i_count
            inv_i_mapping[i_count] = i
            i_count += 1
    return u_mapping, i_mapping, inv_u_mapping, inv_i_mapping


def generate_index_dict(data, u_mapping, i_mapping):
    '''
    Create dictionary of {user_index: item neighbors indices}
    :param data: dataframe of data with attributes such as user_id, item_id
    :param u_mapping: user as key and index as value
    :param i_mapping: item as key and index as value
    :return: bipartite graph user neighbors dictionary
    '''
    data_dict = data.groupby('user_id')['item_id'].apply(list).to_dict()
    index_dict = {}
    
    for user, items in data_dict.items():
        u = u_mapping[user]
        index_dict[u] = [i_mapping[item] for item in items]
    return index_dict


def add_block_to_index_dict(data, data_dict, u_mapping, i_mapping):
    '''
    Add new users and new items to bipartite graph user neighbors dictionary
    :param data: dataframe of data with attributes such as user_id, item_id
    :param data_dict: bipartite graph user neighbors dictionary {user_index: item neighbors indices}
    :param u_mapping: user as key and index as value
    :param i_mapping: item as key and index as value
    :return: updated ipartite graph user neighbors dictionary
    '''
    data_dict = deepcopy(data_dict)
    new_data_dict = data.groupby('user_id')['item_id'].apply(list).to_dict()
    
    for user, items in new_data_dict.items():
        if u_mapping[user] in data_dict:
            data_dict[u_mapping[user]] += [i_mapping[item] for item in items]
        else:
            data_dict[u_mapping[user]] = [i_mapping[item] for item in items]

    return data_dict
    

def convert_dict_to_list(data_dict):
    '''
    Create list of lists of user's item neighbors
    :param data_dict: bipartite graph user neighbors dictionary {user_index: item neighbors indices}
    :return: list of lists of user's item neighbors in the same order as user's indices
    '''
    inner_data_records = []
    for i in range(max(data_dict.keys())+1):
        if i in data_dict:
            inner_data_records.append(data_dict[i].copy())
        else:
            inner_data_records.append([])
    return inner_data_records