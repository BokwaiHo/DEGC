import os
import pandas as pd
import scipy.sparse as sp
from  sklearn.model_selection import train_test_split
import time

from .utils import *
from .preprocessing import *

class Stream_Data(object):
    def __init__(self, dataset, first_segment_time, last_segment_time, shuffle=False, test_ratio = 0.5, valid_test_ratio=0.2, seed=0, replay_ratio=0, sliding_ratio=0):
        self.dataset = dataset
        self.first_segment_time = first_segment_time
        self.last_segment_time = last_segment_time
        self.shuffle = shuffle
        self.valid_test_ratio = valid_test_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.segments = {}
        for i in range(last_segment_time - first_segment_time + 1):
            self.segments[i] = {}

        self.data = self.load_data()
        self.data_size = self.data.shape[0]
        self.latest_size = int(replay_ratio * self.data_size)
        self.sliding_size = int(sliding_ratio * self.data_size)
        
        self.split_and_process_data()
        self.create_train_test_split()

    def load_data(self):
        if self.dataset == 'Alimama':
            loaded_data = load_pickle('data/Alimama/', 'Alimama-nodup-50-50.pkl') 
        elif self.dataset == 'Netflix':
            loaded_data = load_pickle('data/Netflix/', '2000-Netflix-nodup-30-30.pkl')
        elif self.dataset == 'Taobao2014':
            loaded_data = load_pickle('data/Taobao2014/', 'Taobao2014-nodup-10-10.pkl')
        elif self.dataset == 'Taobao2015':
            loaded_data = load_pickle('data/Taobao2015/', 'Taobao2015-nodup-20-20.pkl')
        elif self.dataset == 'Foursquare':
            loaded_data = load_pickle('data/Foursquare/', 'Foursquare-nodup-20-20.pkl')
        else:
            raise NotImplementedError

        return loaded_data

    def split_and_process_data(self):
        acc_index = 0
        last_cur_segment_all = None
        for segment_time in range(self.first_segment_time, self.last_segment_time + 1):
            i = segment_time - self.first_segment_time
            if self.dataset == 'Alimama':
                if segment_time <= 30:
                    cur_segment_all = self.data[(self.data.timestamp >= time.mktime(time.strptime(str('2017-11-{} 00:00:00'.format(segment_time)), "%Y-%m-%d %H:%M:%S"))) \
                        & (self.data.timestamp <= time.mktime(time.strptime(str('2017-11-{} 23:59:59'.format(segment_time)), "%Y-%m-%d %H:%M:%S")))]
                    print('cur_segment_all', cur_segment_all)
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.timestamp >= time.mktime(time.strptime(str('2017-11-{} 00:00:00'.format(segment_time)), "%Y-%m-%d %H:%M:%S"))) \
                        & (self.data.timestamp <= time.mktime(time.strptime(str('2017-11-{} 23:59:59'.format(segment_time)), "%Y-%m-%d %H:%M:%S")))], test_size=self.valid_test_ratio)
                    print('cur_segment_train', cur_segment_train)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.timestamp >= time.mktime(time.strptime(str('2017-11-{} 00:00:00'.format(segment_time)), "%Y-%m-%d %H:%M:%S"))) \
                        & (self.data.timestamp <= time.mktime(time.strptime(str('2017-11-{} 23:59:59'.format(segment_time)), "%Y-%m-%d %H:%M:%S")))])
                    last_acc_index = acc_index
                    acc_index += data_segment_length
                else:
                    cur_segment_all = self.data[(self.data.timestamp >= time.mktime(time.strptime(str('2017-12-{} 00:00:00'.format(segment_time-30)), "%Y-%m-%d %H:%M:%S"))) \
                        & (self.data.timestamp <= time.mktime(time.strptime(str('2017-12-{} 23:59:59'.format(segment_time-30)), "%Y-%m-%d %H:%M:%S")))]
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.timestamp >= time.mktime(time.strptime(str('2017-12-{} 00:00:00'.format(segment_time-30)), "%Y-%m-%d %H:%M:%S"))) \
                        & (self.data.timestamp <= time.mktime(time.strptime(str('2017-12-{} 23:59:59'.format(segment_time-30)), "%Y-%m-%d %H:%M:%S")))], test_size=self.valid_test_ratio)       
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)  
                    data_segment_length = len(self.data[(self.data.timestamp >= time.mktime(time.strptime(str('2017-12-{} 00:00:00'.format(segment_time-30)), "%Y-%m-%d %H:%M:%S"))) \
                        & (self.data.timestamp <= time.mktime(time.strptime(str('2017-12-{} 23:59:59'.format(segment_time-30)), "%Y-%m-%d %H:%M:%S")))])
                    last_acc_index = acc_index
                    acc_index += data_segment_length 

            elif self.dataset == 'Taobao2014':
                first_month = time.strptime(self.data.iloc[0]['timestamp'], '%Y-%m-%d %H').tm_mon
                self.data['month'] = self.data.timestamp.apply(lambda x: int(x[5:7]))
                self.data['day'] = self.data.timestamp.apply(lambda x: int(x[8:10]))
                if segment_time <= 30:
                    cur_segment_all = self.data[(self.data.day == segment_time) & (self.data.month == first_month)]
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.day == segment_time) & (self.data.month == first_month)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.day == segment_time) & (self.data.month == first_month)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length
                else:
                    month = (segment_time - 1) // 30 + first_month
                    day = (segment_time - 1) % 30 +1
                    cur_segment_all = self.data[(self.data.month == month) & (self.data.day == day)]
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.month == month) & (self.data.day == day)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.month == month) & (self.data.day == day)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length

            elif self.dataset == 'Taobao2015':
                first_month = 7
                self.data['month'] = self.data.timestamp.apply(lambda x: int(str(x)[4:6]))
                self.data['day'] = self.data.timestamp.apply(lambda x: int(str(x)[6:8]))
                print('segment_time', segment_time)
                if segment_time <= 31:
                    cur_segment_all = self.data[(self.data.day == segment_time) & (self.data.month == first_month)]
                    if cur_segment_all.empty:
                        print('cur_segment_all is empty')
                        cur_segment_all = last_cur_segment_all
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.day == segment_time) & (self.data.month == first_month)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.day == segment_time) & (self.data.month == first_month)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length
                elif segment_time <= 62:
                    month = 8
                    day = segment_time - 31
                    cur_segment_all = self.data[(self.data.month == month) & (self.data.day == day)]
                    if cur_segment_all.empty:
                        print('cur_segment_all is empty')
                        cur_segment_all = last_cur_segment_all 
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.month == month) & (self.data.day == day)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.month == month) & (self.data.day == day)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length
                elif segment_time <= 92:
                    month = 9
                    day = segment_time - 62
                    cur_segment_all = self.data[(self.data.month == month) & (self.data.day == day)]
                    if cur_segment_all.empty:
                        print('cur_segment_all is empty')
                        cur_segment_all = last_cur_segment_all
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.month == month) & (self.data.day == day)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.month == month) & (self.data.day == day)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length               
                elif segment_time <= 123:
                    month = 10
                    day = segment_time - 92
                    cur_segment_all = self.data[(self.data.month == month) & (self.data.day == day)]
                    if cur_segment_all.empty:
                        print('cur_segment_all is empty')
                        cur_segment_all = last_cur_segment_all
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.month == month) & (self.data.day == day)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.month == month) & (self.data.day == day)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length  
                else:
                    month = 11
                    day = segment_time - 123
                    print('month', month)
                    print('day', day)
                    cur_segment_all = self.data[(self.data.month == month) & (self.data.day == day)]
                    if cur_segment_all.empty:
                        print('cur_segment_all is empty')
                        cur_segment_all = last_cur_segment_all
                    cur_segment_train, cur_segment_val_test = train_test_split(cur_segment_all, test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.month == month) & (self.data.day == day)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length  
                print('cur_segment_all empty is ', cur_segment_all.empty)
                last_cur_segment_all = cur_segment_all

            elif self.dataset == 'Netflix':
                #1999-11-11 2005-12-31#
                first_year = 1999
                self.data['month'] = self.data.timestamp.apply(lambda x: int(str(x)[5:7]))
                self.data['year'] = self.data.timestamp.apply(lambda x: int(str(x)[:4]))
                print('segment_time', segment_time)
                if segment_time <= 12:
                    cur_segment_all = self.data[(self.data.month == segment_time) & (self.data.year == first_year)]
                    if cur_segment_all.empty:
                        print('cur_segment_all is empty')
                        cur_segment_all = last_cur_segment_all
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.month == segment_time) & (self.data.year == first_year)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.month == segment_time) & (self.data.year == first_year)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length
                else:
                    year = (segment_time - 1) // 12 + first_year
                    month = (segment_time - 1) % 12 +1
                    cur_segment_all = self.data[(self.data.month == month) & (self.data.year == year)]
                    if cur_segment_all.empty:
                        print('cur_segment_all is empty')
                        cur_segment_all = last_cur_segment_all 
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.month == month) & (self.data.year == year)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.month == month) & (self.data.year == year)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length
                print('cur_segment_all empty is ', cur_segment_all.empty)
                last_cur_segment_all = cur_segment_all

            elif self.dataset == 'Foursquare':
                #2012-04-03 2014-01-29#
                first_year = 2012
                self.data['month'] = self.data.timestamp.apply(lambda x: int(str(x)[5:7]))
                self.data['year'] = self.data.timestamp.apply(lambda x: int(str(x)[:4]))
                print('segment_time', segment_time)
                if segment_time <= 12:
                    cur_segment_all = self.data[(self.data.month == segment_time) & (self.data.year == first_year)]
                    if cur_segment_all.empty:
                        print('cur_segment_all is empty')
                        cur_segment_all = last_cur_segment_all
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.month == segment_time) & (self.data.year == first_year)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.month == segment_time) & (self.data.year == first_year)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length
                else:
                    year = (segment_time - 1) // 12 + first_year
                    month = (segment_time - 1) % 12 +1
                    cur_segment_all = self.data[(self.data.month == month) & (self.data.year == year)]
                    if cur_segment_all.empty:
                        print('cur_segment_all is empty')
                        cur_segment_all = last_cur_segment_all 
                    cur_segment_train, cur_segment_val_test = train_test_split(self.data[(self.data.month == month) & (self.data.year == year)], test_size=self.valid_test_ratio)
                    cur_segment_val, cur_segment_test = train_test_split(cur_segment_val_test, test_size=self.test_ratio)
                    data_segment_length = len(self.data[(self.data.month == month) & (self.data.year == year)])
                    last_acc_index = acc_index
                    acc_index += data_segment_length
                print('cur_segment_all empty is ', cur_segment_all.empty)
                last_cur_segment_all = cur_segment_all
            else:
                NotImplementedError         
            
            #Train
            if segment_time == self.first_segment_time:
                u_mapping, i_mapping, inv_u_mapping, inv_i_mapping = generate_unique_mappings(cur_segment_train)  
            else:
                u_mapping, i_mapping, inv_u_mapping, inv_i_mapping = update_mappings(cur_segment_train, u_mapping, i_mapping, inv_u_mapping, inv_i_mapping)
            segment_dict_train = generate_index_dict(cur_segment_train, u_mapping, i_mapping)
            self.segments[i]['train'] = convert_dict_to_list(segment_dict_train)
            self.segments[i]['n_user_train'] = len(u_mapping)
            self.segments[i]['n_item_train'] = len(i_mapping)
            for _ in range(self.segments[i]['n_user_train'] - len(self.segments[i]['train'])):
                self.segments[i]['train'].append([])

            #Valid 
            u_mapping, i_mapping, inv_u_mapping, inv_i_mapping = update_mappings(cur_segment_val, u_mapping, i_mapping, inv_u_mapping, inv_i_mapping)
            segment_dict_val = generate_index_dict(cur_segment_val, u_mapping, i_mapping)
            self.segments[i]['val'] = convert_dict_to_list(segment_dict_val)
            self.segments[i]['n_user_val'] = len(u_mapping)
            self.segments[i]['n_item_val'] = len(i_mapping)
            for _ in range(self.segments[i]['n_user_val'] - len(self.segments[i]['val'])):
                self.segments[i]['val'].append([])

            #Test
            u_mapping, i_mapping, inv_u_mapping, inv_i_mapping = update_mappings(cur_segment_test, u_mapping, i_mapping, inv_u_mapping, inv_i_mapping)
            segment_dict_test = generate_index_dict(cur_segment_test, u_mapping, i_mapping)
            self.segments[i]['test'] = convert_dict_to_list(segment_dict_test)
            self.segments[i]['n_user_test'] = len(u_mapping)
            self.segments[i]['n_item_test'] = len(i_mapping)
            for _ in range(self.segments[i]['n_user_test'] - len(self.segments[i]['test'])):
                self.segments[i]['test'].append([])

            #All
            u_mapping, i_mapping, inv_u_mapping, inv_i_mapping = update_mappings(cur_segment_all, u_mapping, i_mapping, inv_u_mapping, inv_i_mapping)
            segment_dict_all = generate_index_dict(cur_segment_all, u_mapping, i_mapping)
            self.segments[i]['all'] = convert_dict_to_list(segment_dict_all)
            self.segments[i]['n_user_all'] = len(u_mapping)
            self.segments[i]['n_item_all'] = len(i_mapping)
            for _ in range(self.segments[i]['n_user_all'] - len(self.segments[i]['all'])):
                self.segments[i]['all'].append([])

            
            if self.latest_size > 0 and i>0:
                reservoir_segment = self.data[last_acc_index - self.latest_size : last_acc_index] if self.latest_size < last_acc_index else self.data[0 : last_acc_index]
                segment_dict_train = generate_index_dict(reservoir_segment, u_mapping, i_mapping)
                self.segments[i]['latest_reservoir'] = convert_dict_to_list(segment_dict_train)
            if self.sliding_size > 0 and i>0:
                reservoir_segment = self.data[last_acc_index - self.sliding_size : last_acc_index] if self.sliding_size < last_acc_index else self.data[0 : last_acc_index]
                segment_dict_train = generate_index_dict(reservoir_segment, u_mapping, i_mapping)
                self.segments[i]['sliding_lists'] = convert_dict_to_list(segment_dict_train)
                self.segments[i]['sliding_matrix'] = generate_sparse_adj_matrix(self.segments[i]['sliding_lists'], self.segments[i-1]['n_user_all'], self.segments[i-1]['n_item_all'])

            

        if self.sliding_size > 0:
            self.segments[0]['latest_reservoir'] = self.segments[1]['latest_reservoir']
            self.segments[0]['sliding_lists'] = self.segments[1]['sliding_lists']
            self.segments[0]['sliding_matrix'] = self.segments[1]['sliding_matrix']
        
        self.u_mapping, self.i_mapping, self.inv_u_mapping, self.inv_i_mapping = u_mapping, i_mapping, inv_u_mapping, inv_i_mapping

        return self.segments


    def create_train_test_split(self):
        for i in range(len(self.segments)):
            if i == 0:
                self.segments[i]['acc_train'] = self.segments[i]['train']
                self.segments[i]['acc_train_plus_val'] = union_lists_of_list(self.segments[i]['acc_train'], self.segments[i]['val'])
                self.segments[i]['acc_train_plus_val_test'] = union_lists_of_list(self.segments[i]['acc_train_plus_val'], self.segments[i]['test'])
            else:
                self.segments[i]['acc_train'] = union_lists_of_list(self.segments[i-1]['acc_train'], self.segments[i]['train'])
                self.segments[i]['acc_train_plus_val'] = union_lists_of_list(self.segments[i]['acc_train'], self.segments[i]['val'])
                self.segments[i]['acc_train_plus_val_test'] = union_lists_of_list(self.segments[i]['acc_train_plus_val'], self.segments[i]['test'])

            self.segments[i]['train_matrix'] = generate_sparse_adj_matrix(self.segments[i]['acc_train'], self.segments[i]['n_user_train'], self.segments[i]['n_item_train'])
            self.segments[i]['val_matrix'] = generate_sparse_adj_matrix(self.segments[i]['acc_train_plus_val'], self.segments[i]['n_user_val'], self.segments[i]['n_item_val'])
            self.segments[i]['test_matrix'] = generate_sparse_adj_matrix(self.segments[i]['acc_train_plus_val_test'], self.segments[i]['n_user_test'], self.segments[i]['n_item_test'])
            self.segments[i]['all_matrix'] = generate_sparse_adj_matrix(self.segments[i]['acc_train_plus_val_test'], self.segments[i]['n_user_test'], self.segments[i]['n_item_test'])
        return self.segments

