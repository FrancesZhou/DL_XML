'''
Created on Jan, 2018

@author: FrancesZhou
'''

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

class NN(object):
    def __init__(self, max_seq_len, vocab_size, word_embedding_dim, label_output_dim, label_prop, num_classify_hidden, args):
        self.max_seq_len = max_seq_len
        self.word_embedding_dim = word_embedding_dim
        self.label_output_dim = label_output_dim
        self.num_classify_hidden = num_classify_hidden
        self.label_prop = tf.constant(label_prop, dtype=tf.float32)
        self.batch_size = args.batch_size
        self.dropout_keep_prob = args.dropout_keep_prob
        #
        self.use_propensity = args.use_propensity
        self.use_comp = args.use_comp
        self.topk = args.topk
        self.factor = args.factor
        #
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        self.neg_inf = tf.constant(value=-np.inf, name='numpy_neg_inf')
        #
        self.word_embedding = tf.get_variable('word_embedding', [vocab_size, word_embedding_dim], initializer=self.weight_initializer)
        #
        self.x_feature_id = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.x_feature_v = tf.placeholder(tf.float32, [None, self.max_seq_len])
        self.y = tf.placeholder(tf.float32, [None, self.label_output_dim])
        self.seqlen = tf.placeholder(tf.int32, [None])

    def competitive_layer(self, y_out, topk=10, factor=0.1):
        x = y_out
        # size: [batch_size, label_output_dim]
        P = (x + tf.abs(x))/2
        values, indices = tf.nn.top_k(P, topk)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk])
        full_indices = tf.stack([my_range_repeated, indices], axis=2)
        full_indices = tf.reshape(full_indices, [-1, 2])
        P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0.,
                                     validate_indices=False)
        pos_value = tf.reduce_sum(P - P_reset, 1, keepdims=True)
        #
        ones_like_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.ones_like(values), [-1]),
                                             default_value=0., validate_indices=False)
        # pos_value: [batch_size, 1]
        # label_prop: [label_output_dim]
        P_tmp = factor * tf.multiply(tf.matmul(pos_value, tf.expand_dims(self.label_prop, axis=0)),
                                     ones_like_reset)
        # p_tmp: [batch_size, label_output_dim]
        # P_reset + p_tmp
        return tf.add(P_reset, P_tmp)

    def build_model(self):
        # x: [batch_size, max_seq_len]
        # y: [batch_size, label_output_dim]
        word_embeddings_padding = tf.concat((tf.constant(0, dtype=tf.float32, shape=[1, self.word_embedding_dim]),
                                            self.word_embedding), axis=0)
        x = tf.nn.embedding_lookup(word_embeddings_padding, self.x_feature_id)
        # x: [batch_size, max_seq_len, word_embedding_dim]
        y = self.y
        # x_emb
        feature_v = tf.layers.batch_normalization(self.x_feature_v)
        #feature_v = tf.layers.dropout(feature_v, rate=self.dropout_keep_prob)
        x_emb = tf.reduce_sum(tf.multiply(x, tf.expand_dims(feature_v, -1)), axis=1)
        # x_emb: [batch_size, word_embedding_dim]
        with tf.name_scope('output'):
            weight_1 = tf.get_variable('weight_1', [self.word_embedding_dim, self.num_classify_hidden],
                                       initializer =self.weight_initializer)
            bias_1 = tf.get_variable('bias_1', [self.num_classify_hidden], initializer=self.const_initializer)
            y_hidden = tf.nn.relu(tf.add(tf.matmul(x_emb, weight_1), bias_1))
            weight_2 = tf.get_variable('weight_2', [self.num_classify_hidden, self.label_output_dim],
                                       initializer=self.weight_initializer)
            #y_out = tf.nn.relu(tf.matmul(y_hidden, weight_2))
            y_out = tf.matmul(y_hidden, weight_2)
            # y_out: [batch_size, label_output_dim]
            # competitive layer
            if self.use_comp:
                y_out = self.competitive_layer(y_out)
        # loss
        if self.use_propensity:
            loss = tf.reduce_sum(
                tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_out), tf.expand_dims(self.label_prop, 0))
            ) + 0.002*tf.nn.l2_loss(weight_1) + 0.002*tf.nn.l2_loss(weight_2)
        else:
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_out))
        return x_emb, tf.sigmoid(y_out), loss

