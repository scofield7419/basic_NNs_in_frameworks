# coding=utf-8

import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_seq2seq
from tensorflow.contrib.seq2seq import *
import numpy as np
from tensorflow.python.layers.core import Dense


class model():
    def __init__(self, get_data_func, epochs, output_max_len):
        self.tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'UNK_TOKEN': 2}
        self.num_tokens = len(self.tokens.keys())
        self.embed_dim = 4
        self.batch_size = 32
        self.num_units = 256
        self.input_vocab_size = 10 + self.num_tokens
        self.output_vocab_size = 10 + self.num_tokens
        self.input_max_len = 100
        self.output_max_len = output_max_len
        self.epochs = epochs
        self.trainImg, trainLabel, self.testImg, testLabel = get_data_func(output_max_len, output_max_len)
        self.trainLabel = np.array(trainLabel) + self.num_tokens
        self.testLabel = np.array(testLabel) + self.num_tokens
        num_train = len(self.trainLabel)
        num_test = len(self.testLabel)
        self.n_per_epoch = num_train // self.batch_size
        self.n_per_epoch_t = num_test // self.batch_size
        # self.sample = self.sampler(num_train)
        # self.sample_t = self.sampler_t(num_test)
        self.is_inference = False
        self.build()

    def set_train_or_inference(self, isTrain=True):
        if isTrain:
            self.is_inference = False
        else:
            self.is_inference = True

    def build(self):
        # define input/output
        self.in_data = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_max_len])
        self.out_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_max_len])
        # end define input

        # define embedding layer
        with tf.variable_scope('embedding'):
            input_embedding_matrix = tf.get_variable(
                'input_embedding_matrix',
                shape=(self.input_vocab_size, self.embed_dim),
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            output_embedding_matrix = tf.get_variable(
                'output_embedding_matrix',
                shape=(self.output_vocab_size, self.embed_dim),
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        with tf.device('/cpu:0'):
            # (batch_size, output_max_len, embed_dim)
            input_embeded = tf.nn.embedding_lookup(input_embedding_matrix, self.in_data)
            output_embeded = tf.nn.embedding_lookup(output_embedding_matrix, self.out_data)

        # end define embedding layer

        # define encoder
        encoder_layer1 = tf.contrib.rnn.GRUCell(num_units=self.num_units)
        encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_layer1, input_embeded, dtype=tf.float32)
        # encoder_output用不上下一步，而是用encoder_final_state
        # end define encoder

        # define decoder
        # define helper for decoder
        if self.is_inference:
            self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
            self.end_token = tf.placeholder(tf.int32, name='end_token')
            helper = GreedyEmbeddingHelper(input_embedding_matrix, self.start_tokens, self.end_token)
        else:
            self.target_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
            self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
            helper = TrainingHelper(output_embeded, self.decoder_seq_length)
        # end define helper for decoder
        # decoder layer
        decoder_layer_1 = tf.contrib.rnn.GRUCell(num_units=self.num_units)
        decoder_layer_2 = Dense(self.output_vocab_size)
        decoder_layer_all = BasicDecoder(
            cell=decoder_layer_1,
            helper=helper,
            initial_state=encoder_final_state,
            output_layer=decoder_layer_2)
        # end decoder layer
        ###这里可增加attention 层
        # decoder run
        self.decoder_out,self.final_state, self.final_sequence_lengths = dynamic_decode(
            decoder=decoder_layer_all,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=self.output_max_len,
            swap_memory=True)
        # end decoder run
        # end define encoder

        # LOSS & OPT
        if not self.is_inference:
            targets = tf.reshape(self.target_ids, [-1])
            logits_flat = tf.reshape(self.decoder_out.rnn_output, [-1, self.output_vocab_size])
            print('shape logits_flat:{}'.format(logits_flat.shape))
            print('shape logits:{}'.format(self.decoder_out.rnn_output.shape))
            self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)
            # define train op
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.prob = tf.nn.softmax(logits)
        # end LOSS & OPT
