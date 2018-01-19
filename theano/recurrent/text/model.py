import os
import pickle
import itertools
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import jieba
import re
import theano
import time
import theano.tensor as T
from gensim.models import Word2Vec
from theano.tensor.shared_randomstreams import RandomStreams
# from train_utils import train_with_validation
from scipy.stats import mode
from theano.tensor.nnet import conv


class DropOutLayer(object):
    def __init__(self, inputs, use_noise, th_rng):
        self.inputs = inputs
        self.outputs = T.switch(use_noise,
                                inputs * th_rng.binomial(inputs.shape, p=0.5, n=1, dtype=inputs.dtype),
                                inputs * 0.5)
        self.params = []

    def save(self, save_to):
        pass


class MeanPoolingLayer(object):
    def __init__(self, inputs, mask):
        '''mean pooling over all words in every sentence.
        inputs: (n_step, batch_size, n_emb)
        mask: (n_step, batch_size)
        '''
        self.inputs = inputs
        self.mask = mask

        self.outputs = T.sum(inputs * mask[:, :, None], axis=0) / T.sum(mask, axis=0)[:, None]
        self.params = []

    def save(self, save_to):
        pass


class MaxPoolingLayer(object):
    def __init__(self, inputs, mask):
        '''max pooling over all words in every sentence.
        inputs: (n_step, batch_size, n_emb)
        mask: (n_step, batch_size)
        '''
        self.inputs = inputs
        self.mask = mask

        self.outputs = T.max(inputs * mask[:, :, None], axis=0)
        self.params = []

    def save(self, save_to):
        pass


class HiddenLayer(object):
    def __init__(self, inputs, activation=T.tanh, load_from=None, rand_init_params=None):
        '''rand_init_params: (rng, (n_in, n_out))
        '''
        self.inputs = inputs
        self.activation = activation

        if load_from is not None:
            W_values = pickle.load(load_from)
            b_values = pickle.load(load_from)
        elif rand_init_params is not None:
            rng, (n_in, n_out) = rand_init_params

            limT = (6 / (n_in + n_out)) ** 0.5
            W_values = rand_matrix(rng, limT, (n_in, n_out))
            if activation is T.nnet.sigmoid:
                W_values *= 4
            b_values = np.zeros(n_out, dtype=theano.config.floatX)
        else:
            raise Exception('Invalid initial inputs!')

        self.W = theano.shared(value=W_values, name='hidden_W', borrow=True)
        self.b = theano.shared(value=b_values, name='hidden_b', borrow=True)

        self.params = [self.W, self.b]

        linear_out = T.dot(inputs, self.W) + self.b
        self.outputs = linear_out if activation is None else activation(linear_out)

    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.b.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)


class EmbLayer(object):
    def __init__(self, inputs, load_from=None, rand_init_params=None, gensim_w2v=None, dic=None):
        '''rand_init_params: (rng, (voc_dim, emb_dim))
        '''
        self.inputs = inputs

        if load_from is not None:
            W_values = pickle.load(load_from)
        elif rand_init_params is not None:
            rng, (voc_dim, emb_dim) = rand_init_params
            W_values = rand_matrix(rng, 1, (voc_dim, emb_dim))

            if gensim_w2v is not None and dic is not None:
                assert gensim_w2v.vector_size == emb_dim

                n_sub = 0
                for idx, word in dic._idx2word.items():
                    if word in gensim_w2v.wv:
                        W_values[idx] = gensim_w2v.wv[word]
                        n_sub += 1
                print('Substituted words by word2vec: %d/%d' % (n_sub, voc_dim))

            W_values = normalize_matrix(W_values)
        else:
            raise Exception('Invalid initial inputs!')

        self.W = theano.shared(value=W_values, name='emb_W', borrow=True)


def th_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)


def rand_matrix(rng, lim, shape, dtype=theano.config.floatX):
    assert lim > 0
    return np.asarray(rng.uniform(low=-lim, high=lim, size=shape), dtype=dtype)


def normalize_matrix(matrix):
    return matrix / np.sum(matrix ** 2, axis=1, keepdims=True) ** 0.5


def get_minibatches_idx(n, batch_size, shuffle=False, rng=None, keep_tail=True):
    idx_seq = np.arange(n, dtype='int32')
    if shuffle:
        if rng is not None:
            rng.shuffle(idx_seq)
        else:
            np.random.shuffle(idx_seq)

    n_batch = n // batch_size
    if n % batch_size > 0:
        n_batch += 1

    batches = []
    for batch_idx in range(n_batch):
        batches.append(idx_seq[(batch_idx * batch_size):((batch_idx + 1) * batch_size)])

    if keep_tail is False and len(batches[-1]) < batch_size:
        del batches[-1]

    return batches


class VotingClassifier(object):
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.n_estimators = len(estimators)

        assert voting in ('hard', 'soft')
        self.voting = voting

    def predict(self, estimator_args, with_prob=False):
        if self.voting == 'hard':
            # sub_res -> (estimator_dim, batch_dim)
            sub_res = np.array([estimator.predict_func(*estimator_args) for estimator in self.estimators],
                               dtype=theano.config.floatX)
            mode_res, count = mode(sub_res, axis=0)
            return (mode_res[0], count[0] / self.n_estimators) if with_prob else mode_res[0]
        else:
            # sub_res -> (estimator_dim, batch_dim, target_dim)
            sub_res = np.array([estimator.predict_prob_func(*estimator_args) for estimator in self.estimators],
                               dtype=theano.config.floatX)
            sub_res = sub_res.mean(axis=0)
            max_res = np.argmax(sub_res, axis=1)
            mean_prob = sub_res[np.arange(sub_res.shape[0]), max_res]
            return (max_res, mean_prob) if with_prob else max_res

    def predict_sent(self, sent, with_prob=False):
        if self.voting == 'hard':
            # sub_res -> (estimator_dim, )
            sub_res = np.array([estimator.predict_sent(sent) for estimator in self.estimators],
                               dtype=np.float32)
            mode_res, count = mode(sub_res)
            return (mode_res[0], count[0] / self.n_estimators) if with_prob else mode_res[0]
        else:
            # sub_res -> (estimator_dim, target_dim)
            sub_res = np.array([estimator.predict_sent(sent, with_prob=True) for estimator in self.estimators],
                               dtype=np.float32)
            sub_res = sub_res.mean(axis=0)
            max_res = np.argmax(sub_res)
            mean_prob = sub_res[max_res]
            return (max_res, mean_prob) if with_prob else max_res


class RNNLayer(object):
    def __init__(self, inputs, mask, load_from=None, rand_init_params=None):
        '''rand_init_params: (rng, (n_in, n_out))
        n_in = emb_dim (* context window size)
        n_out = n_hidden
        '''
        self.inputs = inputs
        self.mask = mask

        if load_from is not None:
            W_values = pickle.load(load_from)
            U_values = pickle.load(load_from)
            b_values = pickle.load(load_from)

            n_out = W_values.shape[1]
        elif rand_init_params is not None:
            rng, (n_in, n_out) = rand_init_params

            limS = 4 * (6 / (n_in + n_out)) ** 0.5

            W_values = rand_matrix(rng, limS, (n_in, n_out))
            U_values = rand_matrix(rng, limS, (n_out, n_out))
            b_values = np.zeros(n_out, dtype=theano.config.floatX)
        else:
            raise Exception('Invalid initial inputs!')

        self.W = theano.shared(value=W_values, name='rnn_W', borrow=True)
        self.U = theano.shared(value=U_values, name='rnn_U', borrow=True)
        self.b = theano.shared(value=b_values, name='rnn_b', borrow=True)

        self.params = [self.W, self.U, self.b]

        def _step(m_t, x_t, h_tm1):
            # hidden units at time t, h(t) is formed from THREE parts:
            #   input at time t, x(t)
            #   hidden units at time t-1, h(t-1)
            #   hidden layer bias, b
            h_t = T.nnet.sigmoid(T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + self.b)
            # mask
            h_t = m_t[:, None] * h_t + (1 - m_t)[:, None] * h_tm1
            return h_t

        n_steps, n_samples, emb_dim = inputs.shape
        hs, updates = theano.scan(fn=_step,
                                  sequences=[mask, inputs],
                                  outputs_info=[T.alloc(np.asarray(0., dtype=theano.config.floatX), n_samples, n_out)])

        self.outputs = hs

    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.U.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.b.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)


class RNNModel(object):
    def __init__(self, corpus, n_emb, n_hidden, pooling, rng=None, th_rng=None,
                 load_from=None, gensim_w2v=None):
        self.corpus = corpus
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.pooling = pooling
        assert pooling in ('mean', 'max')

        if rng is None:
            rng = np.random.RandomState(1226)
        if th_rng is None:
            th_rng = RandomStreams(1226)

        # x/mask: (batch size, nsteps)
        x = T.matrix('x', dtype='int32')
        mask = T.matrix('mask', dtype=theano.config.floatX)
        y = T.vector('y', dtype='int32')
        batch_idx_seq = T.vector('index', dtype='int32')
        use_noise = theano.shared(th_floatX(0.))
        self.x, self.mask, self.y, self.batch_idx_seq, self.use_noise = x, mask, y, batch_idx_seq, use_noise

        # TRANSPOSE THE AXIS!
        trans_x, trans_mask = x.T, mask.T
        # trancate the useless data
        trunc_x, trunc_mask = RNNModel.trunc_inputs_mask(trans_x, trans_mask)
        n_steps, n_samples = trunc_x.shape

        # list of model layers
        model_layers = []
        model_layers.append(EmbLayer(trunc_x, load_from=load_from,
                                     rand_init_params=(rng, (corpus.dic.size, n_emb)),
                                     gensim_w2v=gensim_w2v, dic=corpus.dic))
        model_layers.append(RNNLayer(model_layers[-1].outputs, trunc_mask, load_from=load_from,
                                     rand_init_params=(rng, (n_emb, n_hidden))))
        if pooling == 'mean':
            model_layers.append(MeanPoolingLayer(model_layers[-1].outputs, trunc_mask))
        else:
            model_layers.append(MaxPoolingLayer(model_layers[-1].outputs, trunc_mask))
        model_layers.append(DropOutLayer(model_layers[-1].outputs, use_noise, th_rng))
        model_layers.append(HiddenLayer(model_layers[-1].outputs, activation=T.nnet.softmax, load_from=load_from,
                                        rand_init_params=(rng, (n_hidden, corpus.n_type))))
        self.model_layers = model_layers

        model_params = []
        for layer in model_layers:
            model_params += layer.params

        self.pred_prob = model_layers[-1].outputs
        self.pred = T.argmax(self.pred_prob, axis=1)
        off = 1e-8
        self.cost = -T.mean(T.log(self.pred_prob[T.arange(n_samples), y] + off))

        # attributes with `func` suffix is compiled function
        self.predict_func = theano.function(inputs=[x, mask], outputs=self.pred)
        self.predict_prob_func = theano.function(inputs=[x, mask], outputs=self.pred_prob)

        grads = T.grad(self.cost, model_params)
        self.gr_updates, self.gr_sqr_updates, self.dp_sqr_updates, self.param_updates = ada_updates(model_params, grads)

    def predict_sent(self, sent, with_prob=False):
        idx_seq = self.corpus.dic.sent2idx_seq(sent)

        x = np.array(idx_seq)[None, :]
        mask = np.ones_like(x, dtype=theano.config.floatX)
        if with_prob is False:
            return self.predict_func(x, mask)[0]
        else:
            return self.predict_prob_func(x, mask)[0]

    @staticmethod
    def trunc_inputs_mask(inputs, mask):
        '''keep only the valid steps
        '''
        valid_n_steps = T.cast(T.max(T.sum(mask, axis=0)), 'int32')
        trunc_inputs = inputs[:valid_n_steps]
        trunc_mask = mask[:valid_n_steps]
        return trunc_inputs, trunc_mask

    def save(self, model_fn):
        self.corpus.save(model_fn + '.corpus')
        # do not save rng and th_rng
        with open(model_fn + '.rnn', 'wb') as f:
            pickle.dump(self.n_emb, f)
            pickle.dump(self.n_hidden, f)
            pickle.dump(self.pooling, f)
            for layer in self.model_layers:
                layer.save(f)

    @staticmethod
    def load(model_fn):
        corpus = Corpus.load_from_file(model_fn + '.corpus')
        with open(model_fn + '.rnn', 'rb') as f:
            n_emb = pickle.load(f)
            n_hidden = pickle.load(f)
            pooling = pickle.load(f)
            rnn_model = RNNModel(corpus, n_emb, n_hidden, pooling, load_from=f)
        return rnn_model


class ConvLayer(object):
    def __init__(self, inputs, image_shape, load_from=None, rand_init_params=None):
        '''rand_init_params: (rng, filter_shape)
        inputs: (batch size, stack size, n_words/steps, emb_dim)

        filter_shape: (output stack size, input stack size, filter height, filter width)
            output stack size = ?
            input stack size = 1
            filter height = ?
            filter width = emb_dim (* context window size)

        image_shape(input shape): (batch_size, input stack size, input feature map height, input feature map width)
            batch_size = ?
            input stack size = 1
            input feature map height = n_words/steps
            input feature map width = emb_dim (* context window size)

        output shape: (batch size, output stack size, output feature map height, output feature map width)
            batch_size = ?
            output stack size = ?
            output feature map height = n_words/steps - filter height + 1
            output feature map width = 1
        '''
        self.inputs = inputs

        if load_from is not None:
            W_values = pickle.load(load_from)
            b_values = pickle.load(load_from)

            filter_shape = W_values.shape
        elif rand_init_params is not None:
            rng, filter_shape = rand_init_params
            fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
            fan_out = filter_shape[0] * filter_shape[2] * filter_shape[3]
            limT = (6 / (fan_in + fan_out)) ** 0.5

            W_values = rand_matrix(rng, limT, filter_shape)
            b_values = np.zeros(filter_shape[0], dtype=theano.config.floatX)
        else:
            raise Exception('Invalid initial inputs!')

        self.W = theano.shared(value=W_values, name='conv_W', borrow=True)
        self.b = theano.shared(value=b_values, name='conv_b', borrow=True)
        self.params = [self.W, self.b]

        conv_res = conv.conv2d(input=self.inputs, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
        self.outputs = T.tanh(conv_res + self.b[None, :, None, None])

    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.b.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)


class CNNModel(object):
    def __init__(self, corpus, n_emb, n_hidden, batch_size, conv_size, pooling,
                 rng=None, th_rng=None, load_from=None, gensim_w2v=None):
        '''
        n_hidden: output conv stack size
        conv_size: filter height size
        '''
        self.corpus = corpus
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.conv_size = conv_size
        self.pooling = pooling
        assert pooling in ('mean', 'max')

        if rng is None:
            rng = np.random.RandomState(1226)
        if th_rng is None:
            th_rng = RandomStreams(1226)

        # x/mask: (batch size, nsteps)
        x = T.matrix('x', dtype='int32')
        mask = T.matrix('mask', dtype=theano.config.floatX)
        y = T.vector('y', dtype='int32')
        batch_idx_seq = T.vector('index', dtype='int32')
        use_noise = theano.shared(th_floatX(0.))
        self.x, self.mask, self.y, self.batch_idx_seq, self.use_noise = x, mask, y, batch_idx_seq, use_noise

        # No need for transpose of x/mask in CNN
        n_samples, n_steps = x.shape
        # transpose mask-matrix to be consistent with pooling-layer-inputs
        trans_mask = mask.T
        # truncate mask-matrix to be consistent with conv-outputs
        trunc_mask = trans_mask[(conv_size - 1):]

        # list of model layers
        model_layers = []
        model_layers.append(EmbLayer(x, load_from=load_from,
                                     rand_init_params=(rng, (corpus.dic.size, n_emb)),
                                     gensim_w2v=gensim_w2v, dic=corpus.dic))
        # emb-out: (batch size, n_words/steps, emb_dim)
        # conv-in: (batch size, 1(input stack size), n_words/steps, emb_dim)
        # conv-out: (batch size, n_hidden(output stack size), output feature map height, 1(output feature map width))
        # pooling-in: (output feature map height, batch size, output stack size)
        conv_in = model_layers[-1].outputs[:, None, :, :]
        model_layers.append(ConvLayer(conv_in, image_shape=(batch_size, 1, corpus.maxlen, n_emb), load_from=load_from,
                                      rand_init_params=(rng, (n_hidden, 1, conv_size, n_emb))))
        pooling_in = T.transpose(model_layers[-1].outputs.flatten(3), axes=(2, 0, 1))
        if pooling == 'mean':
            model_layers.append(MeanPoolingLayer(pooling_in, trunc_mask))
        else:
            model_layers.append(MaxPoolingLayer(pooling_in, trunc_mask))
        model_layers.append(DropOutLayer(model_layers[-1].outputs, use_noise, th_rng))
        model_layers.append(HiddenLayer(model_layers[-1].outputs, activation=T.nnet.softmax, load_from=load_from,
                                        rand_init_params=(rng, (n_hidden, corpus.n_type))))
        self.model_layers = model_layers

        model_params = []
        for layer in model_layers:
            model_params += layer.params

        self.pred_prob = model_layers[-1].outputs
        self.pred = T.argmax(self.pred_prob, axis=1)
        off = 1e-8
        self.cost = -T.mean(T.log(self.pred_prob[T.arange(n_samples), y] + off))

        # attributes with `func` suffix is compiled function
        self.predict_func = theano.function(inputs=[x, mask], outputs=self.pred)
        self.predict_prob_func = theano.function(inputs=[x, mask], outputs=self.pred_prob)

        grads = T.grad(self.cost, model_params)
        self.gr_updates, self.gr_sqr_updates, self.dp_sqr_updates, self.param_updates = ada_updates(model_params, grads)

    def predict_sent(self, sent, with_prob=False):
        idx_seq = self.corpus.dic.sent2idx_seq(sent)

        x = np.zeros((self.batch_size, self.corpus.maxlen), dtype='int32')
        x[:, 0:len(idx_seq)] = idx_seq
        mask = np.zeros((self.batch_size, self.corpus.maxlen), dtype=theano.config.floatX)
        mask[:, 0:len(idx_seq)] = 1.
        if with_prob is False:
            return self.predict_func(x, mask)[0]
        else:
            return self.predict_prob_func(x, mask)[0]

    def save(self, model_fn):
        self.corpus.save(model_fn + '.corpus')
        # do not save rng and th_rng
        with open(model_fn + '.cnn', 'wb') as f:
            pickle.dump(self.n_emb, f)
            pickle.dump(self.n_hidden, f)
            pickle.dump(self.batch_size, f)
            pickle.dump(self.conv_size, f)
            pickle.dump(self.pooling, f)
            for layer in self.model_layers:
                layer.save(f)

    @staticmethod
    def load(model_fn):
        corpus = Corpus.load_from_file(model_fn + '.corpus')
        with open(model_fn + '.cnn', 'rb') as f:
            n_emb = pickle.load(f)
            n_hidden = pickle.load(f)
            batch_size = pickle.load(f)
            conv_size = pickle.load(f)
            pooling = pickle.load(f)
            cnn_model = CNNModel(corpus, n_emb, n_hidden, batch_size, conv_size, pooling, load_from=f)
        return cnn_model


class LSTMLayer(object):
    def __init__(self, inputs, mask, load_from=None, rand_init_params=None):
        '''rand_init_params: (rng, (n_in, n_out))
        n_in = emb_dim (* context window size)
        n_out = n_hidden
        '''
        self.inputs = inputs
        self.mask = mask

        if load_from is not None:
            W_values = pickle.load(load_from)
            U_values = pickle.load(load_from)
            b_values = pickle.load(load_from)

            n_out = W_values.shape[1] // 4
        elif rand_init_params is not None:
            rng, (n_in, n_out) = rand_init_params

            limT = (6 / (n_in + n_out * 2)) ** 0.5
            limS = 4 * limT
            # [Wi, Wf, Wo, Wc]
            W_values = rand_matrix(rng, limS, (n_in, 4 * n_out))
            W_values[:, (3 * n_out):(4 * n_out)] /= 4
            # [Ui, Uf, Uo, Uc]
            U_values = rand_matrix(rng, limS, (n_out, 4 * n_out))
            U_values[:, (3 * n_out):(4 * n_out)] /= 4
            # [bi, bf, bo, bc]
            b_values = np.zeros(4 * n_out, dtype=theano.config.floatX)
        else:
            raise Exception('Invalid initial inputs!')

        self.W = theano.shared(value=W_values, name='lstm_W', borrow=True)
        self.U = theano.shared(value=U_values, name='lstm_U', borrow=True)
        self.b = theano.shared(value=b_values, name='lstm_b', borrow=True)

        self.params = [self.W, self.U, self.b]

        def _step(m_t, x_t, h_tm1, c_tm1):
            # x_t is a row of embeddings for several words in same position of different sentences in a minibatch
            # x_t has dimension of (n_samples, n_emb), so it is a matrix
            # m_t is a row of mask matrix, so it is a vector, with dimension of (n_samples, )
            # h_t and c_t are all (n_samples, n_hidden)
            linear_res = T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + self.b

            i_t = T.nnet.sigmoid(linear_res[:, (0 * n_out):(1 * n_out)])
            f_t = T.nnet.sigmoid(linear_res[:, (1 * n_out):(2 * n_out)])
            o_t = T.nnet.sigmoid(linear_res[:, (2 * n_out):(3 * n_out)])
            c_t = T.tanh(linear_res[:, (3 * n_out):(4 * n_out)])

            c_t = f_t * c_tm1 + i_t * c_t
            c_t = m_t[:, None] * c_t + (1 - m_t)[:, None] * c_tm1

            h_t = o_t * T.tanh(c_t)
            h_t = m_t[:, None] * h_t + (1 - m_t)[:, None] * h_tm1
            return h_t, c_t

        n_steps, n_samples, emb_dim = inputs.shape
        (hs, cs), updates = theano.scan(fn=_step,
                                        sequences=[mask, inputs],
                                        outputs_info=[
                                            T.alloc(np.asarray(0., dtype=theano.config.floatX), n_samples, n_out),
                                            T.alloc(np.asarray(0., dtype=theano.config.floatX), n_samples, n_out)])
        self.outputs = hs

    def save(self, save_to):
        pickle.dump(self.W.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.U.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.b.get_value(), save_to, protocol=pickle.HIGHEST_PROTOCOL)


class LSTMModel(object):
    def __init__(self, corpus, n_emb, n_hidden, pooling, rng=None, th_rng=None,
                 load_from=None, gensim_w2v=None):
        self.corpus = corpus
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.pooling = pooling
        assert pooling in ('mean', 'max')

        if rng is None:
            rng = np.random.RandomState(1226)
        if th_rng is None:
            th_rng = RandomStreams(1226)

        # x/mask: (batch size, nsteps)
        x = T.matrix('x', dtype='int32')
        mask = T.matrix('mask', dtype=theano.config.floatX)
        y = T.vector('y', dtype='int32')
        batch_idx_seq = T.vector('index', dtype='int32')
        use_noise = theano.shared(th_floatX(0.))
        self.x, self.mask, self.y = x, mask, y
        self.batch_idx_seq, self.use_noise = batch_idx_seq, use_noise

        # TRANSPOSE THE AXIS!
        trans_x, trans_mask = x.T, mask.T
        # trancate the useless data
        trunc_x, trunc_mask = LSTMModel.trunc_inputs_mask(trans_x, trans_mask)
        n_steps, n_samples = trunc_x.shape

        # list of model layers
        model_layers = []
        model_layers.append(EmbLayer(trunc_x, load_from=load_from,
                                     rand_init_params=(rng, (corpus.dic.size, n_emb)),
                                     gensim_w2v=gensim_w2v, dic=corpus.dic))
        model_layers.append(LSTMLayer(model_layers[-1].outputs, trunc_mask, load_from=load_from,
                                      rand_init_params=(rng, (n_emb, n_hidden))))
        if pooling == 'mean':
            model_layers.append(MeanPoolingLayer(model_layers[-1].outputs, trunc_mask))
        else:
            model_layers.append(MaxPoolingLayer(model_layers[-1].outputs, trunc_mask))
        model_layers.append(DropOutLayer(model_layers[-1].outputs, use_noise, th_rng))
        model_layers.append(HiddenLayer(model_layers[-1].outputs, activation=T.nnet.softmax, load_from=load_from,
                                        rand_init_params=(rng, (n_hidden, corpus.n_type))))
        self.model_layers = model_layers

        model_params = []
        for layer in model_layers:
            model_params += layer.params

        self.pred_prob = model_layers[-1].outputs
        self.pred = T.argmax(self.pred_prob, axis=1)
        off = 1e-8
        self.cost = -T.mean(T.log(self.pred_prob[T.arange(n_samples), y] + off))

        # attributes with `func` suffix is compiled function
        self.predict_func = theano.function(inputs=[x, mask], outputs=self.pred)
        self.predict_prob_func = theano.function(inputs=[x, mask], outputs=self.pred_prob)

        grads = T.grad(self.cost, model_params)
        self.gr_updates, self.gr_sqr_updates, self.dp_sqr_updates, self.param_updates = ada_updates(model_params, grads)

    def predict_sent(self, sent, with_prob=False):
        idx_seq = self.corpus.dic.sent2idx_seq(sent)

        x = np.array(idx_seq)[None, :]
        mask = np.ones_like(x, dtype=theano.config.floatX)
        if with_prob is False:
            return self.predict_func(x, mask)[0]
        else:
            return self.predict_prob_func(x, mask)[0]

    @staticmethod
    def trunc_inputs_mask(inputs, mask):
        '''keep only the valid steps
        '''
        valid_n_steps = T.cast(T.max(T.sum(mask, axis=0)), 'int32')
        trunc_inputs = inputs[:valid_n_steps]
        trunc_mask = mask[:valid_n_steps]
        return trunc_inputs, trunc_mask

    def save(self, model_fn):
        self.corpus.save(model_fn + '.corpus')
        # do not save rng and th_rng
        with open(model_fn + '.lstm', 'wb') as f:
            pickle.dump(self.n_emb, f)
            pickle.dump(self.n_hidden, f)
            pickle.dump(self.pooling, f)
            for layer in self.model_layers:
                layer.save(f)

    @staticmethod
    def load(model_fn):
        corpus = Corpus.load_from_file(model_fn + '.corpus')
        with open(model_fn + '.lstm', 'rb') as f:
            n_emb = pickle.load(f)
            n_hidden = pickle.load(f)
            pooling = pickle.load(f)
            lstm_model = LSTMModel(corpus, n_emb, n_hidden, pooling, load_from=f)
        return lstm_model


class Dictionary(object):
    def __init__(self, word2idx, pat2word=None):
        '''pat2word could be a dict, like {'\d+\.?\d*': '#NUMBER'}
        the dictionary will map words matching the pattern described by the key to its value.
        '''
        self._word2idx = word2idx
        self._idx2word = {idx: w for w, idx in word2idx.items()}
        self._pat2word = pat2word
        self.size = len(self._idx2word)
        assert max(self._idx2word) == self.size - 1
        assert min(self._idx2word) == 0

    def word2idx(self, word):
        if self._pat2word is not None:
            for pat in self._pat2word:
                if re.fullmatch(pat, word):
                    return self._word2idx.get(self._pat2word[pat])
        # idx of 0 is #UNDEF by default
        return self._word2idx.get(word, 0)

    def word_seq2idx_seq(self, word_seq):
        return [self.word2idx(w) for w in word_seq]

    def idx_seq2word_seq(self, idx_seq):
        return [self._idx2word.get(idx, '') for idx in idx_seq]

    def sent2idx_seq(self, sent):
        return self.word_seq2idx_seq(Dictionary.tokenize(sent, lower=True))

    @staticmethod
    def tokenize(sent, lower=True):
        if lower is True:
            sent = sent.lower()
        return [w for w in jieba.cut(sent) if not re.fullmatch('\s+', w)]


class Corpus(object):
    def __init__(self, data_x, data_mask, data_y, word2idx, pat2word=None):
        self.data_x = data_x
        self.data_mask = data_mask
        self.data_y = data_y
        self.size, self.maxlen = data_x.shape
        self.n_type = data_y.max() + 1

        self.dic = Dictionary(word2idx, pat2word=pat2word)

    def train_valid_test(self, valid_ratio=0.15, test_ratio=0.15):
        cut1 = int(self.size * (1 - valid_ratio - test_ratio) + 0.5)
        cut2 = int(self.size * (1 - test_ratio) + 0.5)

        train_x, train_mask, train_y = self.data_x[:cut1], self.data_mask[:cut1], self.data_y[:cut1]
        valid_x, valid_mask, valid_y = self.data_x[cut1:cut2], self.data_mask[cut1:cut2], self.data_y[cut1:cut2]
        test_x, test_mask, test_y = self.data_x[cut2:], self.data_mask[cut2:], self.data_y[cut2:]
        return (train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), (test_x, test_mask, test_y)

    def save(self, corpus_fn):
        with open(corpus_fn, 'wb') as f:
            pickle.dump(self.dic._word2idx, f)
            pickle.dump(self.dic._pat2word, f)
            pickle.dump(self.data_x, f)
            pickle.dump(self.data_mask, f)
            pickle.dump(self.data_y, f)

    @staticmethod
    def load_from_file(fn):
        with open(fn, 'rb') as f:
            word2idx = pickle.load(f)
            pat2word = pickle.load(f)
            data_x = pickle.load(f)
            data_mask = pickle.load(f)
            data_y = pickle.load(f)
        return Corpus(data_x, data_mask, data_y, word2idx, pat2word)

    @staticmethod
    def build_corpus_with_dic(data_x, data_y, maxlen, minlen, dump_to_fn, shuffle=True, pat2word=None):
        '''build corpus and dictionary for raw-corpus

        Args:
            data_x: a numpy.ndarrary type vector, each element of which is a UNTOKENIZED sentence.
            data_y: a numpy.ndarrary type vector, each element of which is a lebel.
        '''
        # special patterns
        word2idx = {'#UNDEF': 0}
        idx = 1
        if pat2word is not None:
            for pat in pat2word:
                word2idx[pat2word[pat]] = idx
                idx += 1

        print('initial data size: %d' % len(data_x))
        # cut sentences and build dic
        new_data_x = []
        new_data_y = []
        for sent, label in zip(data_x, data_y):
            cutted = Dictionary.tokenize(sent)
            # filter by sentence length
            if not (minlen <= len(cutted) <= maxlen):
                continue
            sent_as_idx = []
            for word in cutted:
                if pat2word is not None:
                    for pat in pat2word:
                        if re.fullmatch(pat, word):
                            word = pat2word[pat]
                if word not in word2idx:
                    word2idx[word] = idx
                    idx += 1
                sent_as_idx.append(word2idx[word])
            new_data_x.append(sent_as_idx)
            new_data_y.append(label)

        n_data = len(new_data_x)
        print('filtered data size: %d' % n_data)
        # data_x: 0 is #UNDEF by default
        # data_mask: 0 is masked
        new_data_x_mtx = np.zeros((n_data, maxlen), dtype='int32')
        new_data_mask_mtx = np.zeros((n_data, maxlen), dtype=theano.config.floatX)
        for idx, x in enumerate(new_data_x):
            new_data_x_mtx[idx, :len(x)] = x
            new_data_mask_mtx[idx, :len(x)] = 1.
        new_data_y_vec = np.array(new_data_y)

        print('label description...')
        print(pd.Series(new_data_y_vec).value_counts())

        # shuffle the samples
        if shuffle is True:
            idx_seq = np.arange(n_data)
            np.random.shuffle(idx_seq)
            new_data_x_mtx = new_data_x_mtx[idx_seq]
            new_data_mask_mtx = new_data_mask_mtx[idx_seq]
            new_data_y_vec = new_data_y_vec[idx_seq]

        # dump to file
        with open(dump_to_fn, 'wb') as f:
            pickle.dump(word2idx, f)
            pickle.dump(pat2word, f)
            pickle.dump(new_data_x_mtx, f)
            pickle.dump(new_data_mask_mtx, f)
            pickle.dump(new_data_y_vec, f)


def ada_updates(params, grads, rho=0.95, eps=1e-6):
    '''
    Ada-delta algorithm
    reference: http://www.cnblogs.com/neopenx/p/4768388.html
    '''
    # initialization:
    #   dp    : delta params
    #   dp_sqr: (delta params) ** 2
    #   gr_sqr: gradient ** 2
    running_gr = [theano.shared(p.get_value() * th_floatX(0.)) for p in params]
    running_dp_sqr = [theano.shared(p.get_value() * th_floatX(0.)) for p in params]
    running_gr_sqr = [theano.shared(p.get_value() * th_floatX(0.)) for p in params]
    # update gr
    gr_updates = [(gr_i, new_gr_i) for gr_i, new_gr_i in zip(running_gr, grads)]
    # update gr_sqr
    gr_sqr_updates = [(gr_sqr_i, rho * gr_sqr_i + (1 - rho) * gr_i ** 2) for gr_sqr_i, gr_i in
                      zip(running_gr_sqr, running_gr)]
    # calculate (delta params) by RMS
    # NOTE: here dp_sqr is from last time calculation, because dp has not be calculated!
    dp = [-gr_i * (dp_sqr_i + eps) ** 0.5 / (gr_sqr_i + eps) ** 0.5 for gr_i, dp_sqr_i, gr_sqr_i in
          zip(running_gr, running_dp_sqr, running_gr_sqr)]
    # update dx_sqr
    dp_sqr_updates = [(dp_sqr_i, rho * dp_sqr_i + (1 - rho) * dp_i ** 2) for dp_sqr_i, dp_i in zip(running_dp_sqr, dp)]
    # update params
    param_updates = [(param_i, param_i + dp_i) for param_i, dp_i in zip(params, dp)]

    return gr_updates, gr_sqr_updates, dp_sqr_updates, param_updates


def train_with_validation(train_set, valid_set, corpus,
                          n_hidden=128, n_emb=128, batch_size=32, conv_size=5,
                          pooling_type='mean', model_type='lstm', w2v_fn=None,
                          model_save_fn=None, disp_proc=True):
    '''pooling_type: mean or max
    model_type: lstm, rnn or cnn
    use_w2v: whether to use pre-trained embeddings from word2vec
    '''
    # Only train_set is converted by theano.shared
    train_x, train_mask, train_y = [theano.shared(_) for _ in train_set]
    valid_x, valid_mask, valid_y = valid_set
    n_train, n_valid = len(train_x.get_value()), len(valid_x)

    print("%d training examples" % n_train)
    print("%d validation examples" % n_valid)

    rng = np.random.RandomState(1224)
    th_rng = RandomStreams(1224)

    if model_save_fn is None:
        model_save_fn = os.path.join('model-res', '%s-%s' % (model_type, pooling_type))

    # Load Word2Vec
    if w2v_fn is None:
        gensim_w2v = None
    else:
        print('Loading word2vec model...')
        if not os.path.exists(w2v_fn):
            raise Exception("Word2Vec model doesn't exist!", model_type)
        gensim_w2v = Word2Vec.load(w2v_fn)

    # Define Model
    if model_type == 'lstm':
        model = LSTMModel(corpus, n_emb, n_hidden, pooling_type,
                          rng=rng, th_rng=th_rng, gensim_w2v=gensim_w2v)
    elif model_type == 'rnn':
        model = RNNModel(corpus, n_emb, n_hidden, pooling_type,
                         rng=rng, th_rng=th_rng, gensim_w2v=gensim_w2v)
    elif model_type == 'cnn':
        model = CNNModel(corpus, n_emb, n_hidden, batch_size, conv_size, pooling_type,
                         rng=rng, th_rng=th_rng, gensim_w2v=gensim_w2v)
    else:
        raise Exception("Invalid model type!", model_type)

    x, mask, y = model.x, model.mask, model.y
    batch_idx_seq, use_noise = model.batch_idx_seq, model.use_noise

    f_update_1_gr = theano.function(inputs=[batch_idx_seq],
                                    outputs=model.cost,
                                    updates=model.gr_updates,
                                    givens={x: train_x[batch_idx_seq],
                                            mask: train_mask[batch_idx_seq],
                                            y: train_y[batch_idx_seq]},
                                    on_unused_input='ignore')
    f_update_2_gr_sqr = theano.function(inputs=[], updates=model.gr_sqr_updates)
    f_update_3_dp_sqr = theano.function(inputs=[], updates=model.dp_sqr_updates)
    f_update_4_params = theano.function(inputs=[], updates=model.param_updates)

    # keep validation set consistent
    keep_tail = False if model_type == 'cnn' else True
    valid_idx_batches = get_minibatches_idx(n_valid, batch_size, keep_tail=keep_tail)
    valid_y = np.concatenate([valid_y[idx_batch] for idx_batch in valid_idx_batches])

    # train the model
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    disp_freq = 20
    validation_freq = 100

    max_epoch = 500
    best_iter = 0
    best_validation_err = np.inf

    epoch = 0
    uidx = 0
    done_looping = False
    start_time = time.time()

    while (epoch < max_epoch) and (not done_looping):
        epoch += 1
        # Get new shuffled index for the training set. use rng to make result keep same with specific random-seed
        for idx_batch in get_minibatches_idx(n_train, batch_size, shuffle=True, rng=rng, keep_tail=keep_tail):
            uidx += 1
            use_noise.set_value(1.)

            cost = f_update_1_gr(idx_batch)
            f_update_2_gr_sqr()
            f_update_3_dp_sqr()
            f_update_4_params()

            if uidx % disp_freq == 0 and disp_proc:
                print('epoch %i, minibatch %i, train cost %f' % (epoch, uidx, cost))

            if uidx % validation_freq == 0:
                use_noise.set_value(0.)
                valid_y_pred = [model.predict_func(valid_x[idx_batch], valid_mask[idx_batch]) for idx_batch in
                                valid_idx_batches]
                valid_y_pred = np.concatenate(valid_y_pred)
                this_validation_err = (valid_y_pred != valid_y).mean()
                print('epoch %i, minibatch %i, validation error %f %%' % (epoch, uidx, this_validation_err * 100))

                if this_validation_err < best_validation_err:
                    if this_validation_err < best_validation_err * improvement_threshold:
                        patience = max(patience, uidx * patience_increase)
                    best_validation_err = this_validation_err
                    best_iter = uidx
                    model.save(model_save_fn)

            if patience < uidx:
                done_looping = True
                break

    end_time = time.time()
    print('Optimization complete with best validation score of %f %%, at iter %d' % (
        best_validation_err * 100, best_iter))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, epoch / (end_time - start_time)))


if __name__ == '__main__':
    # TODO: cannot repeat results with same random-seed specified?
    corpus = Corpus.load_from_file(os.path.join('data', 'imdb-prepared.pkl'))

    # ============================= ONE experiment ================================#
    #    train_set, valid_set, test_set = corpus.train_valid_test()
    #    train_with_validation(train_set, valid_set, use_w2v=False)

    # ============================ Cross Validation ===============================#
    data_x, data_mask, data_y = corpus.data_x, corpus.data_mask, corpus.data_y

    #    model_type = 'cnn'
    #    pooling_type = 'max'
    for model_type, pooling_type in itertools.product(['lstm', 'cnn', 'rnn'], ['mean', 'max']):
        save_dn = os.path.join('model-res', 'cv-%s-%s-with-w2v' % (model_type, pooling_type))
        if not os.path.exists(save_dn):
            os.makedirs(save_dn)

        cv_test = KFold(n_splits=5, shuffle=True)
        for test_k, (train_valid_idx, test_idx) in enumerate(cv_test.split(data_x)):
            print('Test fold: %d' % test_k)
            train_valid_x = data_x[train_valid_idx]
            train_valid_mask = data_mask[train_valid_idx]
            train_valid_y = data_y[train_valid_idx]
            test_x = data_x[test_idx]
            test_mask = data_mask[test_idx]
            test_y = data_y[test_idx]

            with open(os.path.join(save_dn, 'test-fold-%d.pkl' % test_k), 'wb') as f:
                pickle.dump(test_x, f)
                pickle.dump(test_mask, f)
                pickle.dump(test_y, f)

            # Use test set as validation, for final model
            train_with_validation((train_valid_x, train_valid_mask, train_valid_y), (test_x, test_mask, test_y), corpus,
                                  pooling_type=pooling_type, model_type=model_type, w2v_fn=r'w2v\enwiki-128.w2v',
                                  model_save_fn=os.path.join(save_dn, 'model-%d' % test_k), disp_proc=False)

            # Split train and validation sets, for performance testing of model
            cv_valid = KFold(n_splits=5, shuffle=True)
            for valid_k, (train_idx, valid_idx) in enumerate(cv_valid.split(train_valid_x)):
                print('Test fold: %d, Valid fold: %d' % (test_k, valid_k))
                train_x = train_valid_x[train_idx]
                train_mask = train_valid_mask[train_idx]
                train_y = train_valid_y[train_idx]
                valid_x = train_valid_x[valid_idx]
                valid_mask = train_valid_mask[valid_idx]
                valid_y = train_valid_y[valid_idx]

                train_with_validation((train_x, train_mask, train_y), (valid_x, valid_mask, valid_y), corpus,
                                      pooling_type=pooling_type, model_type=model_type, w2v_fn=r'w2v\enwiki-128.w2v',
                                      model_save_fn=os.path.join(save_dn, 'model-%d-%d' % (test_k, valid_k)),
                                      disp_proc=False)
