'''
Created on Dec, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

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
    train_label_feature_str = {}
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
        train_label_feature_str[i] = line
        # word
        word_tfidf = {}
        for str in feature_str.split(' '):
            word, tfidf = str.split(':')
            word_tfidf[int(word)] = float(tfidf)
        train_doc_wordID[i] = word_tfidf
    # test
    test_doc_wordID = {}
    test_label = {}
    test_label_feature_str = {}
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
        test_label_feature_str[i] = line
        # word
        word_tfidf = {}
        for str in feature_str.split(' '):
            word, tfidf = str.split(':')
            word_tfidf[int(word)] = float(tfidf)
        test_doc_wordID[i] = word_tfidf
    #all_labels = np.unique(np.concatenate(all_labels)).tolist()
    dump_pickle(train_doc_wordID, args.data + '/trn_tst_data/train_doc_wordID.pkl')
    dump_pickle(train_label, args.data + '/trn_tst_data/train_label.pkl')
    dump_pickle(test_doc_wordID, args.data + '/trn_tst_data/test_doc_wordID.pkl')
    dump_pickle(test_label, args.data + '/trn_tst_data/test_label.pkl')
    #
    dump_pickle(train_label_feature_str, args.data + '/sources/train_feature.pkl')
    dump_pickle(test_label_feature_str, args.data + '/sources/test_feature.pkl')
    train_data_txt_file = args.data + '/sources/train_data.txt'
    with open(train_data_txt_file, 'w') as df:
        df.write('{0} {1} {2}\n'.format(len(train_label_feature_str), fea_num, label_num))
        for _, s_ in train_label_feature_str.items():
            df.write(s_)
    test_data_txt_file = args.data + '/sources/test_data.txt'
    with open(test_data_txt_file, 'w') as df:
        df.write('{0} {1} {2}\n'.format(len(test_label_feature_str), fea_num, label_num))
        for _, s_ in test_label_feature_str.items():
            df.write(s_)
    #return fea_num, label_num, train_doc_wordID, train_label, test_doc_wordID, test_label, train_label_feature_str, test_label_feature_str
    get_label_propensity(args, label_num, train_label, A, B)


# Wikipedia-LSHTC: A=0.5,  B=0.4
# Amazon:          A=0.6,  B=2.6
# Other:		   A=0.55, B=1.5
def get_label_propensity(args, label_num, train_pid_label, A=0.55, B=1.5):
    inv_prop_file = args.data + '/sources/inv_prop.txt'
    train_label = train_pid_label.values()
    train_label = np.concatenate(train_label).tolist()
    label_frequency = dict(Counter(train_label))
    zero_label = set(range(label_num)) - set(label_frequency.keys())
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
    dump_pickle(inv_prop.tolist(), args.data + '/trn_tst_data/inv_prop.pkl')
    inv_prop_dict = dict(zip(labels, inv_prop.tolist()))
    dump_pickle(inv_prop_dict, args.data + '/trn_tst_data/inv_prop_dict.pkl')
    #
    with open(inv_prop_file, 'w') as df:
        df.write(str(inv_prop_dict[labels[0]]))
        for k_ in labels[1:]:
            df.write('\n')
            df.write(str(inv_prop_dict[k_]))
        # for l_, prop_ in inv_prop_dict.items():
        #     df.write(str(l_) + ': ' + str(prop_))
        #     df.write('\n')

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-data', '--data', type=str, default='eurlex', help='which dataset to preprocess')
    args = parse.parse_args()
    if 'amazon' in args.data:
        A, B = (0.6, 2.6)
    elif 'wiki' in args.data:
        A, B = (0.5, 0.4)
    else:
        A, B = (0.55, 1.5)
    get_train_test_data(args, A, B)

if __name__ == "__main__":
    main()
