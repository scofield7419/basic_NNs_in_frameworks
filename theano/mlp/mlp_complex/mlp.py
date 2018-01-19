# -*- coding: utf-8 -*-
from matplotlib import pyplot
import sys
import time
import os
from urllib.request import urlretrieve
import gzip
import pickle
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import classification_report


class LogisticRegression(object):
    """Multi-class Logistic Regression Class == softmax
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                               name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # self.max_prob = self.p_y_given_x[T.arange(input.shape[0]),self.y_pred]

        # self.classify = theano.function(inputs=[input], outputs=[self.y_pred, self.max_prob])

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def pp_errors(self, y, prob, ioi):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        ioi: the index that you are interested in.
        prob: the prob, which is p_y_given_x
        """
        # prob = 0.5
        # ioi = 1
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            # return T.mean(T.neq(self.y_pred, y))
            inprob = self.p_y_given_x[:, ioi]
            pt1 = T.gt(inprob, prob)
            pt2 = T.eq(self.y_pred, ioi)
            pt3 = T.eq(y, ioi)
            ppn = T.sum(pt1 & pt2 & pt3)
            predn = T.sum(pt1 & pt2)
            # return (predn,ppn)
            # return T.sum(T.eq(self.y_pred, y))
            return (ppn, predn)
        else:
            raise NotImplementedError()


def ReLU(x):
    y = T.maximum(0.0, x)
    return (y)


# T.tanh
# T.nnet.sigmoid


class HiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, W=None, b=None, p=1.0,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh
        Hidden unit activation is given by: tanh(dot(input,W) + b)
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :type n_in: int
        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

        self.input = input
        self.p = p
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        output = (lin_output if activation is None
                  else activation(lin_output))
        train_output = output * srng.binomial(size=(n_out,), p=p)
        # train_output=output
        self.output = T.switch(T.neq(is_train, 0), train_output, p * output)

        # parameters of the model
        self.params = [self.W, self.b]

    def drop(self, input_activation):
        """
        :type input: numpy.array
        :param input: layer or weight matrix on which dropout resp. dropconnect is applied

        :type p: float or double between 0. and 1.
        :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.

        """
        mask = self.srng.binomial(n=1, p=self.p, size=input_activation.shape, dtype=theano.config.floatX)
        return input_activation * mask


class MLP(object):
    """Multi-Layer Perceptron Class
    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, is_train, input, n_in, n_hidden, n_out, drop_p=0.5):
        """Initialize the parameters for the multilayer perceptron
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
        :type n_hidden: int
        :param n_hidden: number of hidden units
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """

        # n_hidden2=10
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, is_train=is_train, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh, p=drop_p)

        # self.hiddenLayer2 = HiddenLayer(rng=rng,is_train=is_train, input=self.hiddenLayer.output,
        #                               n_in=n_hidden, n_out=n_hidden2,
        #                               activation=T.tanh, p=drop_p)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                  + abs(self.logRegressionLayer.W).sum() \
            # + abs(self.hiddenLayer2.W).sum()

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                      + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        self.pp_errors = self.logRegressionLayer.pp_errors
        # the parameters of the model are the parameters of the two layer it is
        # made out of

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.max_prob = self.p_y_given_x[T.arange(input.shape[0]), self.y_pred]

        # self.classify = theano.function(inputs=[input], outputs=[self.y_pred, self.max_prob])
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'data/mnist.pkl.gz'


def pickle_load(f, encoding):
    return pickle.load(f, encoding='latin-1')


def _load_data(url=DATA_URL, filename=DATA_FILENAME):
    """Load data from `url` and store the result in `filename`."""
    # print 'filename for the minist datatset:',filename
    if not os.path.exists(filename):
        print("Downloading MNIST dataset")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f, encoding='latin-1')


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = _load_data()
    xs, ys = data[0]
    test_set = (xs[0:10000, :], ys[0:10000])
    # test_set = (xs[400:600,:],ys[400:600]) #check the model fitting.
    # valid_set = (xs[200:400,:],ys[200:400])
    # train_set = (xs[400:numpoint,:],ys[400:numpoint])
    valid_set = test_set
    train_set = (xs[10000:, :], ys[10000:])

    print('shape of x:', train_set[0].shape)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), \
           (test_set_x, test_set_y)


def RMSprop(gparams, params, learning_rate, rho=0.9, epsilon=1e-6):
    """
    param:rho,the fraction we keep the previous gradient contribution
    """
    updates = []
    for p, g in zip(params, gparams):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g))
    return updates


def test_mlp(learning_rate_start=1e-3, learning_rate_end=1e-4, L1_reg=0.000, L2_reg=0.001,
             n_epochs=100, dataset='mnist.pkl.gz', batch_size=200, n_hidden=1000,
             drop_p=0.6, model_err_thresh=0.01, prob_thresh=0.5):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient
    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)
    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
   """
    datasets = load_data()
    # datasets = load_fiidd_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    test_y_numpy = test_set_y.eval()
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    # allocate symbolic variables for the data
    lrlist = numpy.arange(learning_rate_start, learning_rate_end, (learning_rate_end - learning_rate_start) / n_epochs)
    learning_rate = T.scalar('lr')  # learning rate to use
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as ras0terized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels
    is_train = T.iscalar('is_train')  # pseudo boolean for switching between training and prediction
    rng = numpy.random.RandomState(1234)

    input_dimension = train_set_x.get_value(borrow=True).shape[1]
    print('input_dimension:', input_dimension)
    # construct the MLP class
    classifier = MLP(rng=rng, is_train=is_train, input=x, n_in=input_dimension,
                     n_hidden=n_hidden, n_out=10, drop_p=drop_p)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
           + L1_reg * classifier.L1 \
           + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    #    test_model = theano.function(inputs=[index],
    #            outputs=classifier.errors(y),
    #            givens={
    #                x: test_set_x[index * batch_size:(index + 1) * batch_size],
    #                y: test_set_y[index * batch_size:(index + 1) * batch_size],
    #                is_train: 0})

    model_error = theano.function(inputs=[index],
                                  outputs=classifier.errors(y),
                                  givens={
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size],
                                      is_train: numpy.cast['int32'](0)})

    pp_error = theano.function(inputs=[index],
                               outputs=classifier.pp_errors(y, prob_thresh, 1),
                               givens={
                                   x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                   y: test_set_y[index * batch_size:(index + 1) * batch_size],
                                   is_train: numpy.cast['int32'](0)})

    test_prob = theano.function(inputs=[index],
                                outputs=classifier.logRegressionLayer.p_y_given_x,
                                givens={
                                    x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                    is_train: numpy.cast['int32'](0)})

    predict_model = theano.function(inputs=[index],
                                    outputs=classifier.logRegressionLayer.y_pred,
                                    givens={
                                        x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                        is_train: numpy.cast['int32'](0)})

    validate_model = theano.function(inputs=[index],
                                     outputs=classifier.errors(y),
                                     givens={
                                         x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size:(index + 1) * batch_size],
                                         is_train: numpy.cast['int32'](0)})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    # updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    # for param, gparam in zip(classifier.params, gparams):
    #    updates.append((param, param - learning_rate * gparam))
    # using RMSprop(scaling the gradient based on running average)
    updates = RMSprop(gparams, classifier.params, learning_rate)

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index, learning_rate], outputs=cost,
                                  updates=updates,
                                  givens={
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size],
                                      is_train: numpy.cast['int32'](1)})

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    print('num_epoch: %4d' % n_epochs)
    print('num_batch: %4d' % n_train_batches)
    # early-stopping parameters
    patience = 20  # look as this many examples regardless
    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    too_fit = False
    tn_hist = []

    while (epoch < n_epochs) and (True or not done_looping) and (not too_fit):

        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index, lrlist[epoch])
            # iteration number
            # iter = (epoch - 1) * n_train_batches + minibatch_index

        # compute zero-one loss on validation set
        validation_losses = [validate_model(i) for i
                             in range(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)

        model_losses = [model_error(i) for i
                        in range(n_train_batches)]
        this_model_loss = numpy.mean(model_losses)
        tn_hist.append({'train_loss': this_model_loss, 'valid_loss': this_validation_loss})

        pp_losses = [pp_error(i) for i in range(n_valid_batches)]
        prednlist = [te[1] for te in pp_losses]
        ppnlist = [te[0] for te in pp_losses]
        predn = numpy.sum(prednlist)
        ppn = numpy.sum(ppnlist)
        if predn > 0:
            pp = float(ppn) / predn
        else:
            pp = 0.0
        print('e %i, verr %2.1f %% , merr %2.1f %% , pp %i, predn %i ,  pp %2.1f %% \r' \
              % (epoch, this_validation_loss * 100., this_model_loss * 100., ppn, predn, pp * 100.))
        # sys.stdout.flush()
        # print '[learning] epoch %i , model acc %2.2f %%>> '%(epoch,ppn),'completed in %.2f (sec) <<\r'%(time.time()-start_time),
        # sys.stdout.flush()
        epoch = epoch + 1
        if this_model_loss < model_err_thresh:
            too_fit = True

    end_time = time.clock()
    train_loss = numpy.array([i["train_loss"] for i in tn_hist])
    valid_loss = numpy.array([i["valid_loss"] for i in tn_hist])
    pyplot.plot(train_loss, linewidth=2, label="train")
    pyplot.plot(valid_loss, linewidth=2, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    # pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")

    y_pred = []
    # print n_test_batches
    # print list(predict_model(0))
    for i in range(n_test_batches):
        y_pred = y_pred + list(predict_model(i))
    # print 'y_pred', y_pred[0:100]
    # print "Accuracy:", average_precision_score(y_test, y_pred)
    print("Classification report:")
    print(classification_report(test_y_numpy, y_pred))

    pyplot.show()

    print()
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), sys.stderr)
    # visualize the learning process.

    return this_validation_loss, ppn, predn


if __name__ == '__main__':
    test_mlp()
