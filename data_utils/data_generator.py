import os
import pandas as pd
import scipy.sparse as sp

from .utils import *
from .preprocessing import *


class Data(object):
    def __init__(self, dataset, split=[0.5,4,0.1], shuffle=False, split_mode='abs', test_ratio=0.1, seed=0, replay_ratio=0, sliding_ratio=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.test_ratio = test_ratio
        self.seed = seed

        self.split = []
        assert(split[0] + split[2] <= 1)
        if split[1] == 0:
            self.split_interval = 0
        #split : [base_block ratio, number of incremental blocks-1, last_block_ratio]
        else:
            self.split_interval = (1 - split[0] - split[2]) / split[1]
        for i in range(split[1] + 1):
            self.split.append(split[0] + i*self.split_interval)
        self.split.append(1.0)

        self.blocks = {}
        for i in range(len(self.split)):
            self.blocks[i] = {}
        
        self.data = self.load_data()
        self.data_size = self.data.shape[0]

        self.latest_size = int(replay_ratio * self.data_size)
        self.sliding_size = int(sliding_ratio * self.data_size) # with respect to the size of the entire data

        # self.u_mapping, self.i_mapping, self.inv_u_mapping, self.inv_i_mapping = generate_unique_mappings(self.data)
        if split_mode == 'abs':
            self.split_and_process_data_by_time()
        elif split_mode == 'rel':
            self.split_and_process_data_by_relative_time()
        self.create_train_test_split()

    def load_data(self):
        if self.dataset == 'Gowalla-10' or self.dataset == 'gowalla_60':
            loaded_data = load_pickle('data/gowalla/', 'gowalla-no-dup-10-10.pkl')
        elif self.dataset == 'Gowalla-20':
            loaded_data = load_pickle('data/gowalla/', 'no_dup_filtered20_sorted_data.pkl')
        elif self.dataset == 'Taobao2014' or self.dataset == 'tb2014_60':
            loaded_data = load_pickle('data/Taobao2014/', 'Taobao2014-nodup-10-10.pkl')
        elif self.dataset == 'Taobao2015' or self.dataset == 'tb2015_60':
            loaded_data = load_pickle('data/Taobao2015/', 'taobao-2-nodup-10-10-buyonly.pkl')
        elif self.dataset == 'Alimama' or self.dataset == 'almm2017_60':
            loaded_data = load_pickle('data/Alimama/', 'Alimama-nodup-20-20.pkl')
        elif self.dataset == 'lastfm' or self.dataset == 'lastfm_60':
            loaded_data = load_pickle('data/last-fm/', 'lastfm-2k.pkl')
        elif self.dataset == 'lastfm_nodup':
            loaded_data = load_pickle('data/lastfm-nodup/', 'lastfm-2k-nodup-0-0.pkl')
        elif self.dataset == 'yelp_5yrs_60':
            loaded_data = load_pickle('data/yelp/', 'yelp_recent-5yr_10-10.pkl')
        elif self.dataset == 'foursquare':
            loaded_data = load_pickle('data/foursquare/', 'tsmc2019-nodup-10-10.pkl')
        elif self.dataset == 'netflix':
            loaded_data = load_pickle('data/netflix/', 'netflix-nodup-100-30.pkl')
        else:
            raise NotImplementedError
        return loaded_data

    def split_and_process_data_by_relative_time(self):
        '''
        Split each user's item neighbors to several blocks as pre-divided ratio
        :return: dictionary with block as key and corresponding attribute dictionaries of that block
        including list of lists of user's item neighbors, number of items and number of users
        '''
        data_dict = generate_index_dict(self.data, self.u_mapping, self.i_mapping)

        # first block
        cur_block = {}
        for k, v in data_dict.items():
            cur_block[k] = v[:int(self.split[0] * len(v))]
        self.blocks[0]['train'] = convert_dict_to_list(cur_block)
        self.blocks[0]['n_user_train'] = len(self.u_mapping)
        self.blocks[0]['n_item_train'] = len(self.i_mapping)

        for i in range(len(self.split)-1):
            cur_block = {}
            for k, v in data_dict.items():
                cur_block[k] = v[int(len(v)*self.split[i]):int(len(v)*self.split[i+1])]
            self.blocks[i+1]['train'] = convert_dict_to_list(cur_block)
            self.blocks[i+1]['n_user_train'] = len(self.u_mapping)
            self.blocks[i+1]['n_item_train'] = len(self.i_mapping)

        return self.blocks

    def split_and_process_data_by_time(self):
        '''
        Split dataframe (ordered by time) by pre-set ratio
        :return: dictionary with block as key and corresponding attribute dictionaries of that block
        including list of lists of user's item neighbors, number of items and number of users
        '''
        # first block
        cur_block_train = self.data[0:int(self.data_size*self.split[0])]
        u_mapping, i_mapping, inv_u_mapping, inv_i_mapping = generate_unique_mappings(cur_block_train)
        block_dict_train = generate_index_dict(cur_block_train, u_mapping, i_mapping)

        self.blocks[0]['train'] = convert_dict_to_list(block_dict_train)
        self.blocks[0]['n_user_train'] = len(u_mapping)
        self.blocks[0]['n_item_train'] = len(i_mapping)
        
        # rest blocks
        for i in range(len(self.split)-2):
            cur_block_train = self.data[int(self.data_size*self.split[i]):int(self.data_size*self.split[i+1])]
            cur_block_val = self.data[int(self.data_size*self.split[i+1]):int(self.data_size*self.split[i+1] + self.split_interval/2*self.data_size)]
            cur_block_test = self.data[int(self.data_size*self.split[i+1] + self.split_interval/2*self.data_size):int(self.data_size*self.split[i+2])]



            # Train
            u_mapping, i_mapping, inv_u_mapping, inv_i_mapping = update_mappings(cur_block_train, 
                                                                                 u_mapping, 
                                                                                 i_mapping, 
                                                                                 inv_u_mapping, 
                                                                                 inv_i_mapping)
            block_dict_train = generate_index_dict(cur_block_train, u_mapping, i_mapping)
            self.blocks[i+1]['train'] = convert_dict_to_list(block_dict_train)
            self.blocks[i+1]['n_user_train'] = len(u_mapping)
            self.blocks[i+1]['n_item_train'] = len(i_mapping)
            for _ in range(self.blocks[i+1]['n_user_train'] - len(self.blocks[i+1]['train'])):
                self.blocks[i+1]['train'].append([])

            # Val
            u_mapping, i_mapping, inv_u_mapping, inv_i_mapping = update_mappings(cur_block_val, 
                                                                                 u_mapping, 
                                                                                 i_mapping, 
                                                                                 inv_u_mapping, 
                                                                                 inv_i_mapping)
            block_dict_val = generate_index_dict(cur_block_val, u_mapping, i_mapping)
            self.blocks[i+1]['val'] = convert_dict_to_list(block_dict_val)
            self.blocks[i+1]['n_user_val'] = len(u_mapping)
            self.blocks[i+1]['n_item_val'] = len(i_mapping)
            for _ in range(self.blocks[i+1]['n_user_val'] - len(self.blocks[i+1]['val'])):
                self.blocks[i+1]['val'].append([])

            # Test
            u_mapping, i_mapping, inv_u_mapping, inv_i_mapping = update_mappings(cur_block_test, 
                                                                                 u_mapping, 
                                                                                 i_mapping, 
                                                                                 inv_u_mapping, 
                                                                                 inv_i_mapping)
            block_dict_test = generate_index_dict(cur_block_test, u_mapping, i_mapping)
            self.blocks[i+1]['test'] = convert_dict_to_list(block_dict_test)
            self.blocks[i+1]['n_user_test'] = len(u_mapping)
            self.blocks[i+1]['n_item_test'] = len(i_mapping)
            for _ in range(self.blocks[i+1]['n_user_test'] - len(self.blocks[i+1]['test'])):
                self.blocks[i+1]['test'].append([])
           
            if self.latest_size > 0:
                reservoir_block = self.data[int(self.data_size*self.split[i])-self.latest_size:int(self.data_size*self.split[i])]
                block_dict_train = generate_index_dict(reservoir_block, u_mapping, i_mapping)
                self.blocks[i+1]['latest_reservoir'] = convert_dict_to_list(block_dict_train)
            if self.sliding_size > 0:
                reservoir_block = self.data[int(self.data_size*self.split[i])-self.sliding_size:int(self.data_size*self.split[i])]
                block_dict_train = generate_index_dict(reservoir_block, u_mapping, i_mapping)
                self.blocks[i+1]['sliding_lists'] = convert_dict_to_list(block_dict_train)
                self.blocks[i+1]['sliding_matrix'] = generate_sparse_adj_matrix(self.blocks[i+1]['sliding_lists'], self.blocks[i]['n_user_train'], self.blocks[i]['n_item_train'])
        
        # for reservoir initialization
        if self.sliding_size > 0:
            self.blocks[0]['sliding_lists'] = self.blocks[1]['sliding_lists']
            self.blocks[0]['sliding_matrix'] = self.blocks[1]['sliding_matrix']

        self.u_mapping, self.i_mapping, self.inv_u_mapping, self.inv_i_mapping = u_mapping, i_mapping, inv_u_mapping, inv_i_mapping
        return self.blocks

    def create_train_test_split(self):
        #assert self.test_ratio == 0
        #assert self.test_ratio != 0
        for i in range(len(self.blocks)-1):
            if i == 0:
                # no train/val/test seperation for block 0
                self.blocks[i]['acc_train'] = self.blocks[i]['train']
                self.blocks[i]['acc_train_plus_val'] = self.blocks[i]['train']
                self.blocks[i]['acc_train_plus_val_test'] = self.blocks[i]['train']
            else:
                self.blocks[i]['acc_train'] = union_lists_of_list(self.blocks[i-1]['acc_train'], self.blocks[i]['train'])
                self.blocks[i]['acc_train_plus_val'] = union_lists_of_list(self.blocks[i]['acc_train'], self.blocks[i]['val'])
                self.blocks[i]['acc_train_plus_val_test'] = union_lists_of_list(self.blocks[i]['acc_train_plus_val'], self.blocks[i]['test'])

            self.blocks[i]['train_matrix'] = generate_sparse_adj_matrix(self.blocks[i]['acc_train'], self.blocks[i]['n_user_train'], self.blocks[i]['n_item_train'])
            if i > 0:
                self.blocks[i]['val_matrix'] = generate_sparse_adj_matrix(self.blocks[i]['acc_train_plus_val'], self.blocks[i]['n_user_val'], self.blocks[i]['n_item_val'])
                self.blocks[i]['test_matrix'] = generate_sparse_adj_matrix(self.blocks[i]['acc_train_plus_val_test'], self.blocks[i]['n_user_test'], self.blocks[i]['n_item_test'])
        return self.blocks
