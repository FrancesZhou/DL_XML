'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import os
import argparse
import numpy as np
from model.preprocessing.preprocessing import generate_label_embedding_from_file_2
from model.preprocessing.dataloader import *
from model.core.NN import NN
from model.core.solver import ModelSolver
from model.utils.io_utils import load_pickle


def main():
    parse = argparse.ArgumentParser()
    # ---------- environment setting: which gpu -------
    parse.add_argument('-gpu', '--gpu', type=str, default='0', help='which gpu to use: 0 or 1')
    # ---------- foler path of train/test data -------
    parse.add_argument('-valid_labels', '--valid_labels', type=int,
                       default=0, help='-if remove invalid labels')
    parse.add_argument('-data', '--data', type=str, default='eurlex',
                       help='dataset')
    # ---------- model ----------
    parse.add_argument('-word_embedding_dim', '--word_embedding_dim', type=int, default=100, help='dim of word embedding')
    parse.add_argument('-vocab_size', '--vocab_size', type=int, default=5000, help='vocabulary size')
    parse.add_argument('-max_seq_len', '--max_seq_len', type=int, default=500, help='maximum sequence length')
    parse.add_argument('-model', '--model', type=str, default='NN', help='model: NN, LSTM, biLSTM, CNN')
    parse.add_argument('-pretrained_model', '--pretrained_model_path', type=str, default=None, help='path to the pretrained model')
    parse.add_argument('-dropout_keep_prob', '--dropout_keep_prob', type=float,
                       default=0.5, help='keep probability in dropout layer')
    parse.add_argument('-use_sne', '--use_sne', type=int, default=1,
                       help='whether to use sne regularization')
    parse.add_argument('-ac_lbl_ratio', '--ac_lbl_ratio', type=float, default=0.5,
                       help='ratio of active labels in sne regularization')
    parse.add_argument('-use_propensity', '--use_propensity', type=int, default=1,
                       help='whether to use propensity loss')
    parse.add_argument('-use_comp', '--use_comp', type=int, default=0,
                       help='whether to add competitive layer')
    parse.add_argument('-topk', '--topk', type=int, default=10,
                       help='top k neurons in competitive layer')
    parse.add_argument('-factor', '--factor', type=float, default=0.01,
                       help='factor in competitive layer')
    parse.add_argument('-lamb', '--lamb', type=float, default=0.002,
                       help='lambda for weight regularization')
    # ---------- training parameters --------
    parse.add_argument('-n_epochs', '--n_epochs', type=int, default=10, help='number of epochs')
    parse.add_argument('-batch_size', '--batch_size', type=int, default=32, help='batch size for training')
    parse.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='learning rate')
    parse.add_argument('-update_rule', '--update_rule', type=str, default='adam', help='update rule')
    # ------ train or predict -------
    parse.add_argument('-train', '--train', type=int, default=1, help='if training')
    parse.add_argument('-test', '--test', type=int, default=0, help='if testing')
    parse.add_argument('-predict', '--predict', type=int, default=0, help='if predicting')
    args = parse.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print '-------------- load labels ------------------------'
    # default='datasets/eurlex/trn_tst_data/'
    if args.valid_labels:
        args.folder_path = 'datasets/' + args.data + '/trn_tst_data/valid_label_data/'
    else:
        args.folder_path = 'datasets/' + args.data + '/trn_tst_data/all_label_data/'
    #label_prop_dict = load_pickle(args.folder_path + 'inv_prop_dict.pkl')
    label_prop = load_pickle(args.folder_path + 'inv_prop.pkl')
    label_dict = load_pickle(args.folder_path + 'label_dict.pkl')
    all_labels = load_pickle(args.folder_path + 'index_label.pkl')
    num_labels = len(all_labels)
    print 'real number of labels: ' + str(num_labels)
    print 'maximum label: ' + str(np.max(all_labels))
    print 'minimum label: ' + str(np.min(all_labels))
    print 'number of labels: ' + str(num_labels)
    print '-------------- load train/test data -------------------------'
    train_doc = load_pickle(args.folder_path + 'train_doc_wordID.pkl')
    test_doc = load_pickle(args.folder_path + 'test_doc_wordID.pkl')
    train_label = load_pickle(args.folder_path + 'train_label.pkl')
    test_label = load_pickle(args.folder_path + 'test_label.pkl')
    print '============== create train/test data loader ...'
    train_loader = DataLoader_all(train_doc, train_label, label_dict, label_prop,
                                  batch_size=args.batch_size, max_seq_len=args.max_seq_len, ac_lbl_ratio=args.ac_lbl_ratio)
    test_loader = DataLoader_all(test_doc, test_label, label_dict, label_prop,
                                 batch_size=args.batch_size, max_seq_len=args.max_seq_len)
    print '============== build model ...'
    print 'build NN model ...'
    model = NN(args.max_seq_len, args.vocab_size, args.word_embedding_dim, num_labels, label_prop, 32, args)
    args.if_use_seq_len = 1

    print '================= model solver ...'
    solver = ModelSolver(model, train_loader, test_loader,
                         n_epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         update_rule=args.update_rule,
                         learning_rate=args.learning_rate,
                         pretrained_model=args.pretrained_model_path,
                         model_path=args.folder_path + args.model + '/',
                         log_path=args.folder_path + args.model + '/',
                         test_path=args.folder_path + args.model + '/',
                         use_sne=args.use_sne
                         )
    # train
    if args.train:
        print '================= begin training...'
        solver.train(args.folder_path + args.model + '/outcome.txt')

    # test
    if args.test:
        print '================= begin testing...'
        solver.test(args.folder_path + args.model + '/' + args.pretrained_model_path, args.folder_path + args.model + '/test_outcome.txt')

    # predict
    if args.predict:
        print '================= begin predicting...'
        predict_path = args.folder_path+'model_save/'+args.model+'/'
        solver.predict(trained_model_path=predict_path,
                       output_file_path=predict_path+'predict_outcome.txt',
                       k=10, emb_saved=1, can_saved=1)



if __name__ == "__main__":
    main()
