'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from ..utils.op_utils import *

class DataLoader_all():
    def __init__(self, doc_wordID_data, label_data,
                 num_labels, label_prop_dict,
                 batch_size,
                 max_seq_len=5000):
        self.doc_wordID_data = doc_wordID_data
        self.x_feature_indices = {}
        self.x_feature_values = {}
        self.label_data = label_data
        self.label_prop = label_prop_dict
        self.pids = []
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.doc_length = {}
        self.initialize_dataloader()
        self.reset_data()

    def initialize_dataloader(self):
        print 'num of doc: ' + str(len(self.doc_wordID_data))
        print 'num of y: ' + str(len(self.label_data))
        print 'max sequence length: ' + str(self.max_seq_len)
        # label
        zero_prop_label = set(range(self.num_labels)) - set(self.label_prop.keys())
        for zero_l in zero_prop_label:
            self.label_prop[zero_l] = 0
        #
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

    def get_pid_x(self, pool, i, j):
        batch_y = []
        end = min(j, len(pool))
        batch_pid = pool[i:end]
        batch_seq_len = [self.doc_length[p] for p in batch_pid]
        x_feature_id = [self.x_feature_indices[p] for p in batch_pid]
        x_feature_v = [self.x_feature_values[p] for p in batch_pid]
        for pid in batch_pid:
            y = np.zeros(self.num_labels)
            for l in self.label_data[pid]:
                y[l] = 1
            batch_y.append(y)
        # if end < j:
        #     batch_x = np.concatenate((batch_x, np.zeros((j-end, self.max_seq_len), dtype=int)), axis=0)
        #     batch_y = np.concatenate((batch_y, np.zeros((j-end, len(self.all_labels)))), axis=0)
        return batch_pid, x_feature_id, x_feature_v, batch_seq_len, batch_y

    def reset_data(self):
        np.random.shuffle(self.pids)
        self.train_pids, self.val_pids = train_test_split(self.pids, test_size=0.1)
