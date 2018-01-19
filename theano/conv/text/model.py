from theano.tensor.signal import pool
import numpy as np
import theano.tensor as T
import theano
import os
import codecs, random, math
import numpy as np
import theano
import sys
from datetime import datetime


def load_data(corpus_file, dic_file, weights={}, shuffle=False, charset="gb18030"):
    """
    load data form file, each line for a document,
    :param corpus_file,
        format:
        each line for a document or a sentence pair,
            "document or sentence pair: label\tsentence1\tsentence2",
        sentence,
            word1 word2 word3, seperated by space
        when train for classification sentence1 can be same with sentence2,
        when used for prediction, label could be any value,
        when used for language inference, sentence1 is post, sentence2 is response
    :param dic_file:
    :param shuffle:
    :param charset:
    :param weights: class weights, format: "label,weights,count,class_name"
    :return: x1, x2, y, yc(label weights)
    """
    # datas
    data_list = [line.strip() for line in codecs.open(corpus_file, "r", charset)]
    # indices to words
    i2w = [w.strip() for w in codecs.open(dic_file, "r", charset) if w.strip() != ""]
    # words to indices
    w2i = dict([(w, i - 1) for i, w in enumerate(i2w) if w.strip() != ""])

    print("".join(i2w[0:10]).encode(charset))

    # shuffle randomly
    if shuffle:
        random.shuffle(data_list)

    # words to ids
    rx, ry = [], []
    for data in data_list:
        y, x = "0", ""
        if len(data.strip().split("\t")) < 2:
            continue
        parts = data.strip().split("\t")
        [y, x] = parts[0:2]
        rx.append([w2i[x_] for x_ in x.split(" ") if x_ in w2i])
        ry.append(int(y))

    return rx, ry


def format_batch_data(rx, ry, sen_len):
    """
    format word id into numpy matrix, each column for a document
    :param rx:
    :param ry:
    :param sen_len:
    :return: [x, mask_x, y]
    """
    n_samples = len(ry)
    x = np.zeros((sen_len, n_samples)).astype("int32")
    mask_x = np.zeros((sen_len, n_samples)).astype(theano.config.floatX)
    y = np.zeros(n_samples).astype("int32")

    for idx in range(n_samples):
        len_x = sen_len if len(rx[idx]) > sen_len else len(rx[idx])
        x[:len_x, idx] = rx[idx][0:len_x]
        mask_x[:len_x, idx] = 1.

    y = ry

    return [x, mask_x, y]


def test_model(model, test_data, batch_size, sen_len):
    """
    Test model, return(print) precision, recall and f-measure, confuse matrix, loss
    :param model:
    :param test_data:
    :return:
    """
    total_loss = 0.
    total_corr = 0.

    labels = sorted(list(set(test_data[1])))
    classes = len(labels)
    r, p, c = np.zeros(classes), np.zeros(classes), np.zeros(classes)  # real predict correct
    confuse = np.zeros((classes, classes))

    n_samples = len(test_data[1])
    num_batches = n_samples // batch_size
    num_batches = num_batches if num_batches * batch_size == n_samples else num_batches + 1
    for num_batch in range(num_batches):
        # Get a min_batch indices from test_data orderly
        batch_idxs, batch_size = get_min_batch_idxs(len(test_data[0]), num_batch, batch_size, False)
        batch_x = [test_data[0][idx] for idx in batch_idxs]
        batch_y = [test_data[1][idx] for idx in batch_idxs]

        # Format data into numpy matrix
        _x, _mask_x, _y = format_batch_data(batch_x, batch_y, sen_len)

        _p = model.predictions(_x, _mask_x)
        _s = model.weights(_x, _mask_x)
        _c = model.loss(_x, _mask_x, _y)

        for idx in range(batch_size):
            """
            if idx % 3000 == 0 and idx != 0:
                print _s[idx], "\t", _c[idx], "\t", _y[idx], "\t", -math.log(_s[idx][_y[idx]])
            """
            if _y[idx] == _p[idx]:
                c[_y[idx]] += 1.
                total_corr += 1.
            r[_y[idx]] += 1.
            p[_p[idx]] += 1.
            confuse[_p[idx], _y[idx]] += 1.
        total_loss += _c
        del _x, _mask_x, _y

    # test information
    info = str(int(total_corr)) + "Correct, Ratio = " + str(total_corr / n_samples * 100) + "%\n"
    info += datetime.now().strftime("%Y-%m-%d-%H-%M") + "\n"
    for label in labels:
        info += "Label(" + str(label) + "): "
        _p = 0. if p[label] == 0. else c[label] / p[label]
        _r = 0. if r[label] == 0. else c[label] / r[label]
        _f = 0. if _p == 0. or _r == 0. else (_p * _r * 2) / (_p + _r)
        info += "P = " + str(_p) + "%, R = " + str(_r) + "%, F = " + str(_f) + "%\n"
    info += "Confuse Matrix:\n"
    for label in labels:
        info += "\t"
        for value in confuse[label]:
            info += str(int(value)) + " "
        info += "\n"

    info += "Loss:" + str(total_loss) + "\n\n"

    return info


def get_min_batch_idxs(data_size, batch_index, batch_size, random_data=False):
    """
    Get batch_size indices from range(data_size)
    :param data_size.
    :param batch_index, which batch is selected
    :param batch_size.
    :param random_data, if True, example indices will be selected randomly,
           else from begin (batch_index * batch_size) to end (batch_index * batch_size + batch_size or data_size -1).
           if end - begin < batch_size, the rest will be selected randomly
    :return:
    """
    # get begin and end indices
    begin, end = batch_index * batch_size, batch_index * batch_size + batch_size
    if end > data_size: end = data_size
    if end < begin: begin = end

    # get batch index orderly
    idxs = [_ for _ in range(begin, end)]
    # if random_data, get indices randomly
    if random_data: idxs = []
    while len(idxs) < batch_size:
        idxs.append(int(random.random() * data_size) % data_size)

    return idxs, end - begin


def train_model(model, params, argv):
    for i in range(len(argv)):
        print(argv[i])
    data_train_file = argv[2]
    data_test_file = argv[3]
    data_dic_file = argv[4]
    model_save_file = argv[5]

    # Load pre-trained model
    if len(argv) > 6:
        pre_trained_model = argv[6]
        model.load_model(pre_trained_model)
        # print "Load Paraments Done!"
        # print "Test Model's Paramenters"
        # print testModel(model, validate, batch_size, sen_len)

    # Load train and test data
    train_data = load_data(data_train_file, data_dic_file, weights={}, shuffle=True)
    test_data = load_data(data_test_file, data_dic_file, weights={})

    num_batches_seen = 0
    num_batches = len(train_data[1]) // params["batch_size"] + 1

    scale = 1.0
    for epoch in range(params["epoches"]):
        loss_begin, loss_end = [], []
        # For each training example...
        for num_batch in range(num_batches):
            scale *= 0.9995
            batch_idxs, _ = get_min_batch_idxs(len(train_data[0]), num_batch, params["batch_size"], True)
            batch_x = [train_data[0][idx] for idx in batch_idxs]
            batch_y = [train_data[1][idx] for idx in batch_idxs]
            # Format data into numpy matrix
            _x, _mask_x, _y = format_batch_data(batch_x, batch_y, params["sen_len"])

            lr = max(params["learning_rate"] * scale, params["min_learning_rate"])
            loss1 = model.loss(_x, _mask_x, _y)
            model.sgd_step(_x, _mask_x, _y, lr, params["decay"])
            loss2 = model.loss(_x, _mask_x, _y)

            loss_begin.append(loss1)
            loss_end.append(loss2)

            loss1 = sum(loss_begin) / len(loss_begin)
            loss2 = sum(loss_end) / len(loss_end)

            info = "Epoch(%d/%d), Batch(%d/%d), Loss: %f - %f = %f, lr=%f" % (
                epoch, params["epoches"], num_batch, num_batches - 1, loss1, loss2, loss1 - loss2, lr)
            print("\r", info, )
            sys.stdout.flush()
            num_batches_seen += 1
        print()
        print()
        # debug info
        print("num_batches(" + str(num_batches_seen) + "): ")
        print("Epochs(" + str(epoch) + "): ")
        print("\nresult: " + test_model(model, test_data, batch_size=params["batch_size"], sen_len=params["sen_len"]))

        if epoch % params["save_iter"] == params["save_iter"] - 1:
            save_model(model_save_file + "." + str(1001 + epoch)[1:] + ".npz", model)

    save_model(model_save_file + ".final.npz", model)

    return model


def predict(model, params, argv):
    model_file = argv[2]
    data_dic_file = argv[3]
    data_predict_file = argv[4]
    result_save_file = argv[5]

    # Load trained model
    model.load_model(model_file)
    # Load predict data
    predict_data = load_data(data_predict_file, data_dic_file)
    # result save
    sout = open(result_save_file, "w")

    for num_batch in range(len(predict_data[0]) // params["batch_size"] + 1):
        batch_idxs, batch_size = get_min_batch_idxs(len(predict_data[0]), num_batch, params["batch_size"], False)
        batch_x = [predict_data[0][idx] for idx in batch_idxs]
        batch_y = [predict_data[2][idx] for idx in batch_idxs]

        # Format data into numpy matrix
        _x, _mask_x, _y = format_batch_data(batch_x, batch_y, params["sen_len"])

        # predictions
        _p = model.predictions(_x, _mask_x)
        _s = model.weights(_x, _mask_x)

        print("\rpredicted, ", num_batch * params["batch_size"] + batch_size, " ")

        for i in range(0, batch_size):
            info = ''
            for _v in _s[i]:
                info += str(_v) + '\t'
            info += str(_p[i])
            sout.write(info + '\n')

        if batch_size < params["batch_size"]: break

    sout.flush()
    sout.close()


def print_usage():
    print("Usage:")
    print("For training:")
    print(
        "\tpython train.py -t data_train_file data_test_file data_dic_file model_save_file pre_trained_file(optional)")
    print("\tdata_train_file, train data")
    print("\tdata_test_file, test data")
    print()
    print("For prediction:")
    print("\tpython train.py -p model_file dic_file predict_data_file result_save_file")
    print()
    exit(0)


def rms_prop(cost, params_names, params, learning_rate, decay):
    # Gradients:
    # e.g. dE = T.grad(cost, E)
    _grads = [T.grad(cost, params[_n])
              for _i, _n in enumerate(params_names["orign"])]
    # RMSProp caches:
    # e.g. mE = decay * mE + (1 - decay) * dE ** 2
    _caches = [decay * params[_n] + (1 - decay) * _grads[_i] ** 2
               for _i, _n in enumerate(params_names["cache"])]
    # Learning rate:
    # e.g. (E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
    _update_orign = [(params[_n], params[_n] - learning_rate * _grads[_i] / T.sqrt(_caches[_i] + 1e-6))
                     for _i, _n in enumerate(params_names["orign"])]
    # Update cache
    # e.g. (mE, mE)
    _update_cache = [(params[_n], _caches[_i])
                     for _i, _n in enumerate(params_names["cache"])]
    # Merge all updates
    _updates = _update_orign + _update_cache

    return _grads, _updates


def share_params(param_names, params):
    """
    make shared params by theano.shared
    Args:
        param_names:
                 contain 2 types param names, one is orign, the other is cache
                 orign is real parameters that used for this model
                 cache is temporal parameter for rmsprop algorithm
                 both two parameters shared the same shape with parameters in param with idx for id
        params:
                 real parameters that used for making shared parameters
    Returns: shared params
    """
    shared_params = {}
    for _n1, _n2 in zip(param_names["orign"], param_names["cache"]):
        # Theano shared Model's params
        shared_params[_n1] = theano.shared(value=params[_n1].astype(theano.config.floatX), name=_n1)
        # Theano shared Model's params for RMSProp
        shared_params[_n2] = theano.shared(value=np.zeros(params[_n1].shape).astype(theano.config.floatX), name=_n2)
        # print _n1, params[_n1].shape
    return shared_params


def init_weights(shape, low_bound=None, up_bound=None):
    """
    Initialize weights with shape
    Args:
        shape:
        low_bound:
        up_bound:
    Returns:
    """
    if not low_bound or not up_bound:
        low_bound = -np.sqrt(0.01)
        up_bound = -low_bound
    return np.random.uniform(low_bound, up_bound, shape)


def save_model(path, model):
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    if not os.path.exists(path):
        os.mkdir(path)

    sout = open(path + "/setting.txt", "w")
    print(model.hidden_dim, file=sout)
    print(model.embed_dim, file=sout)
    print(model.batch_size, file=sout)
    print(model.output_dim, file=sout)
    print(model.id, file=sout)

    import json

    sout.close()

    for name in model.param_names["orign"]:
        np.save(path + "/" + name + ".npy", model.params[name].get_value())

    print
    ("Saved " + str(model) + " parameters to %s." % path)


def load_model(path, model):
    # sin = open(path + "/setting.txt")
    # model.hidden_dim = int(sin.readline().strip())
    # model.embed_dim = int(sin.readline().strip())
    # model.batch_size = int(sin.readline().strip())
    # model.output_dim = int(sin.readline().strip())

    for name in model.param_names["orign"]:
        model.params[name].set_value(np.load(path + "/" + name + ".npy"))
        print(name, model.params[name].get_value().shape)

    print("Loaded " + str(model) + " parameters from %s." % path)


class ConvPoolLayer:
    def __init__(self, batch_size, sen_len, embed_dim,
                 filter_size=10,
                 filter_shape=(10, 1, 2, 2),
                 channels=1,
                 pooling_mode="max",
                 id=""):
        """
        define Convolutional Layers, with filter_size filters with shape (filter_shape)
        :param filter_size:
        :param filter_shape:
        :param channels:
        conv_w is a 4-D tensor with shape(filter_size, channels, filter_height, filter_width)
        conv_b is a vector with shape(filter_size)
        """
        self.batch_size = batch_size

        self.sen_len = sen_len
        self.embed_dim = embed_dim
        self.input_shape = (batch_size, channels, sen_len, embed_dim)
        self.pooling_shape = (sen_len - filter_shape[2] + 1, embed_dim - filter_shape[3] + 1)

        self.filter_size = filter_size
        self.filter_shape = filter_shape

        self.channels = channels
        self.pooling_mode = pooling_mode
        self.id = id

        params = dict()
        params[id + "conv_w"] = init_weights(self.filter_shape)
        params[id + "conv_b"] = init_weights((filter_size,))

        self.param_names = {
            "orign": [id + "conv_w", id + "conv_b"],
        }
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = share_params(self.param_names, params)

        import json
        print(
            "ConvLayer Build! Params = %s" % (
                json.dumps(
                    {"id": id,
                     "batch_size": batch_size,
                     "sen_len": sen_len,
                     "embed_dim": embed_dim,
                     "filter_size": filter_size,
                     "filter_shape": filter_shape,
                     "channels": channels}, indent=4)
            ))

    def get_output(self, input):
        """
        input is
        :param input: A 4-D tensor with shape(batch_size, channels, sen_len, embedding_size),
                      usually, embedding_size == filter_width
        :return: A 4-D tensor with shape(batch_size, filter_size, sen_len-filter_height+1, embedding_size-filter_width+1)
        """
        # usually output is a 4-D tensor with shape(batch_size, filters, sen_len-filter_height+1, 1)
        output = T.nnet.conv2d(input=input,
                               filters=self.params[self.id + "conv_w"],
                               input_shape=self.input_shape,
                               filter_shape=self.filter_shape,
                               border_mode="valid")
        #  output = output.reshape([self.batch_size, self.filter_size, self.pooling_shape[0], self.pooling_shape[1]])
        # add a bias to each filter
        output += self.params[self.id + "conv_b"].dimshuffle("x", 0, "x", "x")

        if self.pooling_mode != "average":  # self.pooling_mode == "max":
            output = pool.pool_2d(input=output,
                                  ignore_border=True,
                                  ds=self.pooling_shape,
                                  st=self.pooling_shape,
                                  padding=(0, 0),  # padding shape
                                  mode="max")
            # output = theano.printing.Print("Conv Pool Out")(output)
            return output.flatten().reshape([self.batch_size, self.filter_size])
        elif self.pooling_mode == "average":
            output = pool.pool_2d(input=output,
                                  ignore_border=True,
                                  ds=self.pooling_shape,
                                  st=self.pooling_shape,
                                  padding=(0, 0),  # padding shape
                                  mode="average_inc_pad")

            return output.flatten().reshape([self.batch_size, self.filter_size])


class EmbddingLayer:
    def __init__(self, feature_dim, embed_dim, id="", w2v_file="", dic_file=""):
        """
        embedding layer
        Args:
            feature_dim: vocabulary size
            input_dim: embedding dimension
            id:
            w2v_file:
            dic_file:
        """
        self.output_dim = feature_dim
        self.embed_dim = embed_dim
        self.id = id

        params = dict()
        params[id + "word_embed"] = init_weights((feature_dim, embed_dim))

        # load pre-trained word vector for initialization
        if w2v_file != "" and dic_file != "":
            dic = {w.strip(): i for i, w in enumerate(open(dic_file))}
            sin = open(w2v_file)
            sin.readline()
            for line in sin:
                parts = line.strip().split(" ")
                w = parts[0]
                if w not in dic:
                    continue
                value = [float(_) for _ in parts[1:]]
                if len(value) != embed_dim:
                    break
                params[id + "word_embed"][dic[w], 0:embed_dim] = value[0:embed_dim]

        # normalization word vector
        for i in range(params[id + "word_embed"].shape[0]):
            base = np.sum(params[id + "word_embed"][i, :] * params[id + "word_embed"][i, :])
            base = np.sqrt(base)
            params[id + "word_embed"][i, :] /= base

        # define parameters' names
        self.param_names = {"orign": [id + "word_embed"]}
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = share_params(self.param_names, params)
        import json
        print("Embdding Layer Build! Params = %s" % (
            json.dumps(
                {"id": id,
                 "embed_dim": embed_dim,
                 "feature_dim": feature_dim}, indent=4)
        ))

        def embedding(self, input):
            """
            embedding an input word by word
            Args:
                input: a tensor.fmatrix with shape (len_idx,)
            Returns: a tensor.ftensor3 with shap (len_idx, embed_dim)
            """
            return self.params[self.id + "word_embed"][input, :]


class PredictLayer:
    def __init__(self, hidden_dim, output_dim, drop=True, id=""):
        """
        Predict words probability depends on hidden states.
        Functions:
            predict_weights_real(input), predict_idx_real(input):
                used for predict result when training is done
            predict_weights(input), predict_idx(input):
                used for training that might be predict with dropout for hidden
        Args:
            hidden_dim:
            output_dim: can be number of words or classes
            drop: weather used drop when predict
            id:
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.id = id

        params = dict()
        params[id + "full_w"] = init_weights((hidden_dim, output_dim))
        params[id + "full_b"] = init_weights((output_dim))

        self.param_names = {
            "orign": [id + "full_w", id + "full_b"],
        }
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = share_params(self.param_names, params)

        if drop:
            self.rng = np.random.RandomState(3435)
        import json
        print("BasicMLPLayer Build! Params = %s" % (
            json.dumps(
                {"id": id,
                 "hidden": hidden_dim,
                 "output_dim": output_dim,
                 "drop": drop}, indent=4)
        ))

        def get_l1_cost(self):
            return T.sum(T.abs(self.params[self.id + "full_w"]))

        def get_l2_cost(self):
            return T.sum(self.params[self.id + "full_w"] ** 2)

        def predict(self, input, drop=True, train=True):
            """
            Calculate predcit based on input
            Args:
                input:
                    a theano.tensor.fmatrix with shape (len_idx, hidden_dim)
                drop:
                    whether use dropout when make prediction
            Returns:
                weights, a theano.tensor.fmatrix with shape (len_idx, output_dim)
                predict, a theano.tensor.ivector with shap (len_idx)
            """
            params = self.params
            id = self.id
            scale = 1.0
            # drop is used for training
            if drop and train:
                srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(999999))
                # p=1-p because 1's indicate keep and p is prob of dropping
                mask = srng.binomial(n=1, p=0.5, size=(input.shape[0], input.shape[1]))
                # The cast is important because
                # int * float32 = float64 which pulls things off the gpu
                input = input * T.cast(mask, theano.config.floatX)
            elif drop and not train:
                # when dropout is used for training process,
                # input should be timed with (1-dropratio), except bais
                scale *= 0.5

            weights = input.dot(params[id + "full_w"]) * scale + params[id + "full_b"]
            weights = T.nnet.softmax(weights)
            predict = T.argmax(weights, axis=1)

            return weights, predict


class CNN:
    def __init__(self, settings, id=""):
        """
        Layers:
            1, Embedding Layer
            2, {ConvLayer + PoolLayer} * K
            3, MLPLayers
            4, PredictLayer
        """
        self.id = id
        self.batch_size = settings["batch_size"]
        self.sen_len = settings["sen_len"]
        self.feature_dim = settings["feature_dim"]
        self.embed_dim = settings["embed_dim"]
        # [(10, 1, 3, 3), (), ()]
        self.filters_shapes = settings["filters_shapes"]
        self.pooling = settings["pooling"]
        # input size = \sigma_i^k{kernel.shape[0]}
        self.hidden_dim = -1
        self.hidden_dims = [sum([shape[0] for shape in self.filters_shapes])] + settings["hidden_dims"]
        self.output_dim = settings["output_dim"]
        self.drop = settings["drop"]

        self.word_embedding_layer = EmbddingLayer(
            feature_dim=self.feature_dim,
            embed_dim=self.embed_dim,
            id=self.id + "word_embed_"
        )

        if len(self.hidden_dims) >= 2:
            self.mlp_layers = MLPLayers(
                hidden_dims=self.hidden_dims,
                nonlinear=T.nnet.sigmoid,
                id=self.id + "mlp_"
            )
        else:
            self.mlp_layers = None

        self.conv_layers = []
        for idx, conv_shape in enumerate(self.filters_shapes):
            conv_layer = ConvPoolLayer(batch_size=self.batch_size,
                                       sen_len=self.sen_len,
                                       embed_dim=self.embed_dim,
                                       filter_size=conv_shape[0],
                                       channels=conv_shape[1],
                                       filter_shape=conv_shape,
                                       pooling_mode=self.pooling,
                                       id=self.id + "conv_" + str(idx) + "_")
            self.conv_layers.append(conv_layer)

        self.predict_layer = PredictLayer(
            hidden_dim=self.hidden_dims[-1],
            output_dim=self.output_dim,
            id=self.id + "predict_"
        )

        import json

        # collects all parameters
        self.params = dict()
        self.param_names = {"orign": [], "cache": []}
        self.params.update(self.word_embedding_layer.params)
        if self.mlp_layers:
            self.params.update(self.mlp_layers.params)
        for conv_layer in self.conv_layers:
            self.params.update(conv_layer.params)
        self.params.update(self.predict_layer.params)

        for name in ["orign", "cache"]:
            self.param_names[name] += self.word_embedding_layer.param_names[name]
            if self.mlp_layers:
                self.param_names[name] += self.mlp_layers.param_names[name]
            for conv_layer in self.conv_layers:
                self.param_names[name] += conv_layer.param_names[name]
            self.param_names[name] += self.predict_layer.param_names[name]

        param_key = self.params.keys()
        param_key.sort()
        putty = json.dumps(param_key, indent=4)
        print("param_names = ", putty)

        self.__build__model__()

    def __build__model__(self):
        x = T.imatrix('x')  # first sentence
        x_mask = T.fmatrix("x_mask")
        y = T.ivector('y')  # label

        # for compatibility, input's shape is (sen_len, batch_size)
        # for cnn convolution, the input shape should be (batch_size, 1, sen_len, embed_dim)
        # embedding encoder and decoder inputs with two embedding layers
        embedding = self.word_embedding_layer.embedding(x.flatten())
        embedding = embedding.reshape([x.shape[0], 1, x.shape[1], self.embed_dim])
        embedding = embedding.dimshuffle(2, 1, 0, 3)
        # embedding = embedding.reshape([input.shape[0], input.shape[1], self.embed_dim])

        conv_outs = [conv_layer.get_output(embedding) for conv_layer in self.conv_layers]
        conv_hidden = T.concatenate(conv_outs, axis=1)
        # conv_hidden = theano.printing.Print("Conv Hidden")(conv_hidden)

        final_hidden_states = self.mlp_layers.get_output(conv_hidden)

        train_weights, _ = self.predict_layer.predict(final_hidden_states, drop=self.drop, train=True)
        # calculate cross entropy
        cost = T.sum(T.nnet.categorical_crossentropy(train_weights, y.flatten()))

        cost += self.predict_layer.get_l2_cost() * 0.01

        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        grads, updates = rms_prop(cost, self.param_names, self.params, learning_rate, decay)

        # Assign functions
        self.loss = theano.function([x, x_mask, y], cost, on_unused_input='ignore')
        self.bptt = theano.function([x, x_mask, y], grads, on_unused_input='ignore')
        self.sgd_step = theano.function(
            [x, x_mask, y, learning_rate, decay],
            updates=updates, on_unused_input='ignore')

        # for test
        weights, predictions = self.predict_layer.predict(final_hidden_states, drop=self.drop, train=False)
        self.weights = theano.function([x, x_mask], weights, on_unused_input='ignore')
        self.predictions = theano.function([x, x_mask], predictions, on_unused_input='ignore')


class MLPLayer:
    def __init__(self, hidden_dim, output_dim, drop=True, id=""):
        """
        Predict words probability depends on hidden states.
        Args:
            hidden_dim:
            output_dim: can be number of words or classes
            drop: weather used drop when predict
            id:
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.id = id

        params = dict()
        params[id + "full_w"] = init_weights((hidden_dim, output_dim))
        params[id + "full_b"] = init_weights((output_dim))

        self.param_names = {
            "orign": [id + "full_w", id + "full_b"],
        }
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = share_params(self.param_names, params)

        if drop:
            self.rng = np.random.RandomState(3435)
        import json
        print("PredictLayer Build! Params = %s" % (
            json.dumps(
                {"id": id,
                 "hidden": hidden_dim,
                 "output_dim": output_dim,
                 "drop": drop}, indent=4)
        ))

        def get_output(self, input, drop=True, train=True):
            """
            Calculate predcit based on input
            Args:
                input:
                    a theano.tensor.fmatrix with shape (len_idx, hidden_dim)
                drop:
                    whether use dropout when make prediction
                train:
                    different strategy for dropout when training and predicting
            Returns:
                weights, a theano.tensor.fmatrix with shape (len_idx, output_dim)
                predict, a theano.tensor.ivector with shap (len_idx)
            """
            params = self.params
            id = self.id
            scale = 1.0
            # drop is used for training
            if drop and train:
                srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(999999))
                # p=1-p because 1's indicate keep and p is prob of dropping
                mask = srng.binomial(n=1, p=0.5, size=(input.shape[0], input.shape[1]))
                # The cast is important because
                # int * float32 = float64 which pulls things off the gpu
                input = input * T.cast(mask, theano.config.floatX)
            elif drop and not train:
                scale = 0.5  # multi drop ratio

            weights = input.dot(params[id + "full_w"]) * scale + params[id + "full_b"]

            return weights


class MLPLayers:
    def __init__(self, hidden_dims=[300, 300, 300], nonlinear=T.nnet.sigmoid, drop=True, id=""):
        """
        :param input_dim:
        :param layer_shapes[embed_dim, hidden_layer, ..., output_dim]:
        :param id:
        """
        self.layer_shapes = hidden_dims
        self.id = id
        self.nonlinear = nonlinear
        # define e
        # whether used a third part embedding

        self.layers = []
        cnt = 1
        for i, o in zip(hidden_dims[:-1], hidden_dims[1:]):
            layer = MLPLayer(hidden_dim=i, output_dim=o, id=self.id + str(1000 + cnt), drop=False)
            self.layers.append(layer)
            cnt += 1

        # collects all parameters
        self.params = dict()
        self.param_names = {"orign": [], "cache": []}
        for layer in self.layers:
            self.params.update(layer.params)

        for name in ["orign", "cache"]:
            for layer in self.layers:
                self.param_names[name] += layer.param_names[name]

        import json
        print("Build MLPModel Done! Params = %s" % (json.dumps({"id": id, "layer_shapes": hidden_dims})))

    def get_output(self, input):
        """
        input with shape (batch_size, layer_shapes[0])
        :param input:
        :return: shape (batch_size, layer_shapes[-1])
        """
        next_input = input
        for layer in self.layers:
            next_input = layer.get_output(next_input, drop=False)
            # next_input = self.nonlinear(next_input)

        return next_input


if __name__ == "__main__":
    # from train_utils import *

    if len(sys.argv) <= 1 or (sys.argv[1] != "-t" and sys.argv[1] != "-p"):
        print_usage()
        exit(0)

    params = {
        "output_dim": 2,
        "feature_dim": -1,
        "embed_dim": 64,
        "filters_shapes": [(10, 1, 2, 64), (10, 1, 3, 64), (10, 1, 5, 64), (10, 1, 4, 64)],
        "pooling": "max",
        "hidden_dims": [256, 256],
        "sen_len": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "min_learning_rate": 1e-6,
        "decay": 0.95,
        "epoches": 200,
        "save_iter": 2,
        "drop": True
    }

    dic_file = sys.argv[4]
    if sys.argv[1] == "-p": dic_file = sys.argv[3]
    feature_dim = len([_ for _ in open(dic_file)])
    params["feature_dim"] = feature_dim

    import json

    print("Settings =", json.dumps(params, indent=2))

    model_cnn = CNN(params, id="CNN_")

    if sys.argv[1] == "-t":
        train_model(model_cnn, params, sys.argv)
    elif sys.argv[1] == "-p":
        predict(model_cnn, params, sys.argv)
