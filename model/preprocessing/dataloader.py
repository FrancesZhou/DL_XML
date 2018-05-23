'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
#from scipy.sparse import csr_matrix
from ..utils.op_utils import *
from ..utils.io_utils import *

class DataLoader_all():
    def __init__(self, doc_wordID_data, label_data,
                 label_dict, label_prop,
                 batch_size,
                 max_seq_len=5000,
                 ac_lbl_ratio=0.5,
                 folder_path=''):
        self.doc_wordID_data = doc_wordID_data
        self.x_feature_indices = {}
        self.x_feature_values = {}
        self.label_data = label_data
        self.label_prop = np.array(label_prop, dtype=np.float32)
        self.pids = []
        self.label_dict = label_dict
        #print label_dict
        self.num_labels = len(self.label_dict.keys())
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.ac_lbl_ratio = ac_lbl_ratio
        self.doc_length = {}
        self.initialize_dataloader()
        self.reset_data()
        self.datapath = folder_path

    def initialize_dataloader(self):
        print 'num of doc: ' + str(len(self.doc_wordID_data))
        print 'num of y: ' + str(len(self.label_data))
        print 'max sequence length: ' + str(self.max_seq_len)
        # label
        self.pids = np.asarray(self.label_data.keys())
        for pid in self.pids:
            temp = sorted(self.doc_wordID_data[pid].items(), key=lambda e: e[1], reverse=True)
            temp2 = sorted(temp[:self.max_seq_len], key=lambda e: e[0], reverse=False)
            feature_id, feature_v = zip(*temp2)
            seq_len = min(len(feature_id), self.max_seq_len)
            feature_indices = np.array(list(feature_id) + (self.max_seq_len - seq_len) * [0])
            feature_indices[:seq_len] = feature_indices[:seq_len] + 1
            self.x_feature_indices[pid] = feature_indices
            self.x_feature_values[pid] = np.array(list(feature_v) + (self.max_seq_len - seq_len) * [0])
            self.doc_length[pid] = seq_len
            #
            # labels

    def get_pid_x(self, pool, i, j):
        batch_y = []
        end = min(j, len(pool))
        batch_pid = pool[i:end]
        # batch_seq_len = [self.doc_length[p] for p in batch_pid]
        x_feature_id = [self.x_feature_indices[p] for p in batch_pid]
        x_feature_v = [self.x_feature_values[p] for p in batch_pid]
        for pid in batch_pid:
            y = np.zeros(self.num_labels)
            for l in self.label_data[pid]:
                y[self.label_dict[l]] = 1
            batch_y.append(y)
        # if end < j:
        #     batch_x = np.concatenate((batch_x, np.zeros((j-end, self.max_seq_len), dtype=int)), axis=0)
        #     batch_y = np.concatenate((batch_y, np.zeros((j-end, len(self.all_labels)))), axis=0)
        return batch_pid, x_feature_id, x_feature_v, batch_y

    def reset_data(self):
        np.random.shuffle(self.pids)
        self.train_pids, self.val_pids = train_test_split(self.pids, test_size=0.1)

    def distance_i_j(self, v_i, v_j):
        intersect_ind = np.intersect1d(v_i, v_j)
        minus_dis = np.sum(self.label_prop[intersect_ind])
        return (self.prop_dis-minus_dis)/self.num_labels

    def get_distance_matrix(self):
        self.index_pids = np.array(self.label_data.keys())
        pid_num = len(self.index_pids)
        row_ind = [[k]*len(self.label_data[self.index_pids[k]]) for k in range(pid_num)]
        row_ind = np.concatenate(row_ind)
        col_ind = []
        for p_ in self.index_pids:
            col_ind.append([self.label_dict[l_] for l_ in self.label_data[p_]])
        col_ind = np.concatenate(col_ind)
        data = np.ones_like(row_ind)
        pid_label_matrix = csc_matrix((data, (row_ind, col_ind)), shape=(pid_num, self.num_labels))
        #dis_matrix = csc_matrix((pid_num, pid_num), dtype=np.float32)
        dis_matrix = lil_matrix((pid_num, pid_num), dtype=np.float32)
        self.pid_dis = {}
        # self.pid_pid_dis = []
        label_sort = np.argsort(-self.label_prop)[:int(self.num_labels*self.ac_lbl_ratio)]
        #
        self.prop_dis = np.sum(self.label_prop)
        sigma = 1
        print 'calculate hamming distance'
        #print pid_num
        for l_ind in label_sort:
            ac_pid_index = pid_label_matrix[:, l_ind].nonzero()[0]
            # calculate the distance between these pids
            for i in range(len(ac_pid_index)):
                for j in range(i+1, len(ac_pid_index)):
                    pid_i = ac_pid_index[i]
                    pid_j = ac_pid_index[j]
                    #print pid_i
                    #print pid_j
                    v_i = [self.label_dict[l_] for l_ in self.label_data[self.index_pids[pid_i]]]
                    v_j = [self.label_dict[l_] for l_ in self.label_data[self.index_pids[pid_j]]]
                    dis_i_j = self.distance_i_j(v_i, v_j)
                    dis_matrix[pid_i, pid_j] = np.exp(-dis_i_j*dis_i_j/(2*sigma*sigma))
                    dis_matrix[pid_j, pid_i] = dis_matrix[pid_i, pid_j]
        print 'done'
        print 'calculate probability'
        dis_matrix = dis_matrix.tocsc()
        for nz_index in set(dis_matrix.nonzero()[1]):
            dis_matrix[:, nz_index] = dis_matrix[:, nz_index]/np.sum(dis_matrix[:, nz_index])
        print 'done'
        print 'calculate symmetrical probability'
        for nz_index in zip(*dis_matrix.nonzero()):
            nz_row, nz_col = nz_index
            #print nz_row
            #print nz_col
            p_i_j = (dis_matrix[nz_row, nz_col] + dis_matrix[nz_col, nz_row])/2
            try:
                self.pid_dis[self.index_pids[nz_row]][self.index_pids[nz_col]] = p_i_j
            except KeyError:
                self.pid_dis[self.index_pids[nz_row]] = {}
                self.pid_dis[self.index_pids[nz_row]][self.index_pids[nz_col]] = p_i_j
            try:
                self.pid_dis[self.index_pids[nz_col]][self.index_pids[nz_row]] = p_i_j
            except KeyError:
                self.pid_dis[self.index_pids[nz_col]] = {}
                self.pid_dis[self.index_pids[nz_col]][self.index_pids[nz_row]] = p_i_j
        print 'done'
        self.pid_dis_keys = self.pid_dis.keys()
        dump_pickle(self.pid_dis, self.datapath + 'pid_dis.pkl')
        # np.random.shuffle(self.pid_pid_dis)

    def load_distance_matrix(self):
        self.pid_dis = load_pickle(self.datapath + 'pid_dis.pkl')
        self.pid_dis_keys = self.pid_dis.keys()

    def get_pid_pid_dis(self, pid):
        #pid = self.index_pids[i]
        p1_f_id = np.expand_dims(self.x_feature_indices[pid], axis=0)
        p1_f_v = np.expand_dims(self.x_feature_values[pid], axis=0)
        p2_id, p2_dis = zip(*(self.pid_dis[pid].items()))
        p2_f_id = [self.x_feature_indices[p_] for p_ in p2_id]
        p2_f_v = [self.x_feature_values[p_] for p_ in p2_id]
        return np.array(p1_f_id, np.int32), np.array(p1_f_v, np.float32), \
               np.array(p2_f_id, np.int32), np.array(p2_f_v, np.float32), \
               np.array(p2_dis, np.float32)

