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
        self.lamb = args.lamb
        #
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        self.neg_inf = tf.constant(value=-np.inf, name='numpy_neg_inf')
        #
        with tf.name_scope('word_embedding'):
            self.word_embedding = tf.get_variable('word_embedding', [vocab_size, word_embedding_dim], initializer=self.weight_initializer)
            self.variable_summaries(self.word_embedding)
        with tf.name_scope('weight_1'):
            self.weight_1 = tf.get_variable('weight_1', [self.word_embedding_dim, self.num_classify_hidden],
                                            initializer=self.weight_initializer)
            self.variable_summaries(self.weight_1)
        with tf.name_scope('bias_1'):
            self.bias_1 = tf.get_variable('bias_1', [self.num_classify_hidden], initializer=self.const_initializer)
            self.variable_summaries(self.bias_1)
        with tf.name_scope('weight_2'):
            self.weight_2 = tf.get_variable('weight_2', [self.num_classify_hidden, self.label_output_dim],
                                            initializer=self.weight_initializer)
            self.variable_summaries(self.weight_2)
        #
        self.x_feature_id = tf.placeholder_with_default(tf.constant(0, dtype=tf.int32, shape=[1, self.max_seq_len]),
                                                        [None, self.max_seq_len])
        self.x_feature_v = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[1, self.max_seq_len]),
                                                       [None, self.max_seq_len])
        self.y = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[1, self.label_output_dim]),
                                             [None, self.label_output_dim])
        # self.seqlen = tf.placeholder(tf.int32, [None])
        #
        self.p1_f_id = tf.placeholder_with_default(tf.constant(0, dtype=tf.int32, shape=[1, self.max_seq_len]),
                                                   [1, self.max_seq_len])
        self.p1_f_v = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[1, self.max_seq_len]),
                                                  [1, self.max_seq_len])
        self.p2_f_id = tf.placeholder_with_default(tf.constant(0, dtype=tf.int32, shape=[1, self.max_seq_len]),
                                                   [None, self.max_seq_len])
        self.p2_f_v = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[1, self.max_seq_len]),
                                                  [None, self.max_seq_len])
        self.p1_p2_dis = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[1]), [None])
        #self.training = tf.placeholder(tf.bool, shape=())

    #def hidden_competitive_layer(self):

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def competitive_layer(self, y_out, topk=100, factor=0):
        x = y_out
        # size: [batch_size, label_output_dim]
        #P = (x + tf.abs(x))/2
        P = y_out
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
        feature_v = self.x_feature_v
        y = self.y
        with tf.name_scope('word_embedding'):
            word_embeddings_padding = tf.concat((tf.constant(0, dtype=tf.float32, shape=[1, self.word_embedding_dim]),
                                                 self.word_embedding), axis=0)
            x = tf.nn.embedding_lookup(word_embeddings_padding, self.x_feature_id)
            # x: [batch_size, max_seq_len, word_embedding_dim]
            # x_emb
            #feature_v = tf.layers.batch_normalization(feature_v, training=self.training)
            #feature_v = tf.layers.dropout(feature_v, rate=self.dropout_keep_prob, training=self.training)
            x_emb = tf.reduce_sum(tf.multiply(x, tf.expand_dims(feature_v, -1)), axis=1)
            # x_emb: [batch_size, word_embedding_dim]
            tf.summary.histogram('x_emb', x_emb)
        with tf.name_scope('output'):
            y_hidden = tf.nn.relu(tf.add(tf.matmul(x_emb, self.weight_1), self.bias_1))
            tf.summary.histogram('y_hidden', y_hidden)
            # BN and dropout
            #y_hidden = tf.layers.batch_normalization(y_hidden, training=self.training)
            #y_hidden = tf.layers.dropout(y_hidden, rate=self.dropout_keep_prob, training=self.training)
            #
            #y_out = tf.nn.relu(tf.matmul(y_hidden, weight_2))
            y_out_hidden = tf.matmul(y_hidden, self.weight_2)
            #y_out = tf.sigmoid(y_out)
            #self.variable_summaries(y_out)
            tf.summary.histogram('y_out', y_out_hidden)
            # y_out: [batch_size, label_output_dim]
            # competitive layer
            if self.use_comp:
                with tf.name_scope('competitve_layer'):
                    y_out_1 = self.competitive_layer(y_out_hidden, self.topk, self.factor)
                    #self.variable_summaries(y_out)
                    tf.summary.histogram('comp_out', y_out_1)
                    eps = tf.constant(value=np.finfo(float).eps, dtype=tf.float32, name='numpy_eps')
                    y_out = tf.where(tf.equal(y_out_1, 0), tf.ones_like(y_out_1) * eps, tf.sigmoid(y_out_1))
                    #y_out = tf.where(tf.greater(y_out, 0), tf.sigmoid(y_out), tf.ones_like(y_out)*eps)
            else:
                y_out_1 = y_out_hidden
                y_out = y_out_hidden
        with tf.name_scope('loss'):
            # loss
            if self.use_propensity:
                crs_entrpy = tf.add(tf.multiply(y, tf.log(y_out)), tf.multiply(1-y, tf.log(1-y_out)))
                loss = -tf.reduce_sum(tf.multiply(crs_entrpy, tf.expand_dims(self.label_prop, 0)))
                #loss = tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_out), tf.expand_dims(self.label_prop, 0))) \
                #       + 3*tf.nn.l2_loss(self.weight_1) + 2*tf.nn.l2_loss(self.weight_2)
                # self.lamb*tf.nn.l2_loss(weight_1) + self.lamb*tf.nn.l2_loss(weight_2)
                # loss = -tf.reduce_sum(tf.multiply(tf.add(tf.multiply(y, tf.log(y_out)), tf.multiply(1-y, tf.log(1-y_out))), tf.expand_dims(self.label_prop, 0)))
            else:
                loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_out))
        tf.summary.scalar('loss', loss)
        return x_emb, y_out, loss, tf.nn.l2_loss(self.weight_1), tf.nn.l2_loss(self.weight_2), tf.reduce_sum(tf.where(tf.equal(y_out_1, 0), tf.zeros_like(y_out_1), tf.ones_like(y_out_1)))

    def t_sne(self):
        # x_1: [1, max_seq_len]
        # x_2: [all, max_seq_len]
        with tf.name_scope('x_emb'):
            word_embeddings_padding = tf.concat((tf.constant(0, dtype=tf.float32, shape=[1, self.word_embedding_dim]),
                                                 self.word_embedding), axis=0)
            # x: [none, max_seq_len, word_embedding_dim]
            # x_emb: [none, word_embedding_dim]
            x_1 = tf.nn.embedding_lookup(word_embeddings_padding, self.p1_f_id)
            x_1_emb = tf.reduce_sum(tf.multiply(x_1, tf.expand_dims(self.p1_f_v, -1)), axis=1)
            x_2 = tf.nn.embedding_lookup(word_embeddings_padding, self.p2_f_id)
            x_2_emb = tf.reduce_sum(tf.multiply(x_2, tf.expand_dims(self.p2_f_v, -1)), axis=1)
            # q_i_j
        with tf.name_scope('distance'):
            dis = tf.reciprocal(tf.norm(x_1_emb - x_2_emb, axis=1) + 1)
            dis = tf.divide(dis, tf.reduce_sum(dis))
            tf.summary.histogram('dis', dis)
        with tf.name_scope('t_sne_loss'):
            loss = tf.reduce_sum(tf.multiply(self.p1_p2_dis, tf.log(self.p1_p2_dis/dis)))
            tf.summary.scalar('t_sne_loss', loss)
        return loss




