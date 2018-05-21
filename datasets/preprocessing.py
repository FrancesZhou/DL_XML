'''
Created on Dec, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import numpy as np
import math
import cPickle as pickle
from collections import Counter

# data_source_path = 'sources'
# data_des_path = 'trn_tst_data'

def dump_pickle(data, file):
    try:
        with open(file, 'w') as datafile:
            pickle.dump(data, datafile)
    except Exception as e:
        raise e

def load_pickle(file):
    try:
        with open(file, 'r') as datafile:
            data = pickle.load(datafile)
    except Exception as e:
        raise e
    return data

def write_file(data, file):
    try:
        with open(file, 'w') as datafile:
            for line in data:
                datafile.write(str(line[0]) + '\t' + str(line[1]) + '\n')
    except Exception as e:
        raise e

def load_txt(file):
    try:
        with open(file, 'r') as df:
            data = df.readlines()
    except Exception as e:
        raise e
    return data

def get_train_test_data(args, A, B):
    train_label_fea_file = '{0}/sources/xml/{1}_train.txt'.format(args.data, args.data)
    test_label_fea_file = '{0}/sources/xml/{1}_test.txt'.format(args.data, args.data)
    train_label_fea = load_txt(train_label_fea_file)
    head_str = train_label_fea[0]
    _, fea_num, label_num = head_str.split(' ')
    fea_num = int(fea_num)
    label_num = int(label_num)
    train_label_fea = train_label_fea[1:]
    test_label_fea = load_txt(test_label_fea_file)[1:]
    all_labels = []
    # train
    train_doc_wordID = {}
    train_label = {}
    #train_label_feature_str = {}
    train_feature_str = {}
    for i in xrange(len(train_label_fea)):
        line = train_label_fea[i]
        labels_str, feature_str = line.split(' ', 1)
        # label
        labels_str = labels_str.split(',')
        try:
            labels = [int(label) for label in labels_str]
        except ValueError:
            continue
        all_labels.append(labels)
        train_label[i] = labels
        # feature
        #train_label_feature_str[i] = line
        train_feature_str[i] = feature_str
        # word
        word_tfidf = {}
        for str in feature_str.split(' '):
            word, tfidf = str.split(':')
            word_tfidf[int(word)] = float(tfidf)
        train_doc_wordID[i] = word_tfidf
    # test
    test_doc_wordID = {}
    test_label = {}
    #test_label_feature_str = {}
    test_feature_str = {}
    for i in xrange(len(test_label_fea)):
        line = test_label_fea[i]
        labels_str, feature_str = line.split(' ', 1)
        # label
        labels_str = labels_str.split(',')
        try:
            labels = [int(label) for label in labels_str]
        except ValueError:
            continue
        test_label[i] = labels
        # feature
        test_feature_str[i] = feature_str
        #test_label_feature_str[i] = line
        # word
        word_tfidf = {}
        for str in feature_str.split(' '):
            word, tfidf = str.split(':')
            word_tfidf[int(word)] = float(tfidf)
        test_doc_wordID[i] = word_tfidf

    if args.valid_labels:
        all_labels = np.unique(np.concatenate(all_labels)).sort()
        label_num = len(all_labels)
        print 'number of valid labels: {0}'.format(label_num)
        label_dict = dict(zip(all_labels, np.arange(label_num)))
        # get valid train/test data
        for p_, l_ in train_label.items():
            l_ = np.intersect1d(l_, all_labels)
            if len(l_):
                train_label[p_] = l_
            else:
                del train_label[p_]
                del train_doc_wordID[p_]
                del train_feature_str[p_]
        for p_, l_ in test_label.items():
            l_ = np.intersect1d(l_, all_labels)
            if len(l_):
                test_label[p_] = l_
            else:
                del test_label[p_]
                del test_doc_wordID[p_]
                del test_feature_str[p_]
        train_data_txt_file = args.data + '/sources/valid_label_data/train_data.txt'
        test_data_txt_file = args.data + '/sources/valid_label_data/test_data.txt'
        trn_tst_data_dir = 'valid_label_data/'
    else:
        all_labels = np.arange(label_num)
        label_dict = dict(zip(all_labels, all_labels))
        train_data_txt_file = args.data + '/sources/all_label_data/train_data.txt'
        test_data_txt_file = args.data + '/sources/all_label_data/test_data.txt'
        trn_tst_data_dir = 'all_label_data/'
        # dump_pickle(train_label_feature_str, args.data + '/sources/train_feature.pkl')
        # dump_pickle(test_label_feature_str, args.data + '/sources/test_feature.pkl')
    #
    # for baseline data
    with open(train_data_txt_file, 'w') as df:
        df.write('{0} {1} {2}\n'.format(len(train_feature_str), fea_num, label_num))
        for p_, s_ in train_feature_str.items():
            l_ = [label_dict[k] for k in train_label[p_]]
            l_str = ''
            for l_i in l_[:-1]:
                l_str = l_str + str(l_i) + ','
            l_str = l_str + str(l_[-1]) + ' '
            # add features
            l_str = l_str + s_
            df.write(l_str)
    with open(test_data_txt_file, 'w') as df:
        df.write('{0} {1} {2}\n'.format(len(test_feature_str), fea_num, label_num))
        for p_, s_ in test_feature_str.items():
            l_ = [label_dict[k] for k in test_label[p_]]
            l_str = ''
            for l_i in l_[:-1]:
                l_str = l_str + str(l_i) + ','
            l_str = l_str + str(l_[-1]) + ' '
            # add features
            l_str = l_str + s_
            df.write(l_str)
            df.write(s_)
    #return fea_num, label_num, train_doc_wordID, train_label, test_doc_wordID, test_label, train_label_feature_str, test_label_feature_str
    sort_labels, _ = zip(*sorted(label_dict.items(), key=lambda e: e[1]))
    get_label_propensity(args, trn_tst_data_dir, sort_labels, train_label, A, B)
    #
    dump_pickle(label_dict, args.data + '/trn_tst_data/' + trn_tst_data_dir + 'label_dict.pkl')
    dump_pickle(sort_labels, args.data + '/trn_tst_data/' + trn_tst_data_dir + 'index_label.pkl')
    #
    dump_pickle(train_doc_wordID, args.data + '/trn_tst_data/' + trn_tst_data_dir + 'train_doc_wordID.pkl')
    dump_pickle(train_label, args.data + '/trn_tst_data/' + trn_tst_data_dir + 'train_label.pkl')
    dump_pickle(test_doc_wordID, args.data + '/trn_tst_data/' + trn_tst_data_dir + 'test_doc_wordID.pkl')
    dump_pickle(test_label, args.data + '/trn_tst_data/' + trn_tst_data_dir + 'test_label.pkl')



# Wikipedia-LSHTC: A=0.5,  B=0.4
# Amazon:          A=0.6,  B=2.6
# Other:		   A=0.55, B=1.5
def get_label_propensity(args, dir_name, sort_labels, train_pid_label, A=0.55, B=1.5):
    inv_prop_file = args.data + '/sources/' + dir_name + 'inv_prop.txt'
    train_label = train_pid_label.values()
    train_label = np.concatenate(train_label).tolist()
    label_frequency = dict(Counter(train_label))
    zero_label = set(sort_labels) - set(label_frequency.keys())
    zero_dict = dict(zip(zero_label, np.zeros(len(zero_label), dtype=np.float32)))
    label_frequency.update(zero_dict)
    #
    labels, fre = zip(*sorted(label_frequency.items(), key=lambda e: e[0], reverse=False))
    #labels, fre = zip(*label_frequency.iteritems())
    fre = np.array(fre)
    #
    N = len(train_pid_label)
    C = (math.log(N)-1) * (B + 1)**A
    inv_prop = 1 + C * (fre + B)**(-A)
    #
    #dump_pickle(inv_prop.tolist(), args.data + '/trn_tst_data/inv_prop.pkl')
    inv_prop_dict = dict(zip(labels, inv_prop.tolist()))
    dump_pickle(inv_prop_dict, args.data + '/trn_tst_data/' + dir_name + 'inv_prop_dict.pkl')
    #
    # sorted(self.doc_wordID_data[pid].items(), key=lambda e: e[1], reverse=True)
    #sort_labels, _ = zip(*sorted(label_dict.items(), key=lambda e: e[1]))
    inv_prop_sort_labels = [inv_prop_dict[l_] for l_ in sort_labels]
    dump_pickle(inv_prop_sort_labels, args.data + '/trn_tst_data/' + dir_name + 'inv_prop.pkl')
    with open(inv_prop_file, 'w') as df:
        df.write(str(inv_prop_sort_labels[0]))
        for k_ in inv_prop_sort_labels[1:]:
            df.write('\n')
            df.write(str(k_))
        # df.write(str(inv_prop_dict[labels[0]]))
        # for k_ in labels[1:]:
        #     df.write('\n')
        #     df.write(str(inv_prop_dict[k_]))

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-data', '--data', type=str, default='eurlex', help='which dataset to preprocess')
    parse.add_argument('-valid_labels', '-valid_labels', type=bool, default=True, help='if remove invalid labels')
    args = parse.parse_args()
    if args.valid_labels:
        if not os.path.exists(args.data + '/sources/valid_label_data/'):
            os.makedirs(args.data + '/sources/valid_label_data/')
        if not os.path.exists(args.data + '/trn_tst_data/valid_label_data/'):
            os.makedirs(args.data + '/trn_tst_data/valid_label_data/')
    else:
        if not os.path.exists(args.data + '/sources/all_label_data/'):
            os.makedirs(args.data + '/sources/all_label_data/')
        if not os.path.exists(args.data + '/trn_tst_data/all_label_data/'):
            os.makedirs(args.data + '/trn_tst_data/all_label_data/')
    if 'amazon' in args.data:
        A, B = (0.6, 2.6)
    elif 'wiki' in args.data:
        A, B = (0.5, 0.4)
    else:
        A, B = (0.55, 1.5)
    get_train_test_data(args, A, B)

if __name__ == "__main__":
    main()
