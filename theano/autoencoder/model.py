import numpy
import json

import theano
import theano.tensor as T


class Network(object):
    def __init__(self, new_network=True, **kwargs):
        if new_network:
            self.new_network(**kwargs)

    def new_network(self, layers, input, rng=numpy.random.RandomState(), bias=False, activation="tanh",
                    output_activation="linear"):
        hidden_layer = []
        for layer in range(len(layers) - 2):
            hiddenLayer = Layer(
                name=('hidden%d') % (layer),
                rng=rng,
                input=input,
                n_in=layers[layer],
                n_out=layers[layer + 1],
                bias=bias,
                activation=activation
            )
            hidden_layer.append(hiddenLayer)
            input = hidden_layer[layer].output

        output_layer = Layer(
            name="output",
            rng=rng,
            input=hidden_layer[-1].output,
            n_in=layers[-2],
            n_out=layers[-1],
            bias=bias,
            activation=output_activation
        )
        self.create_network(hidden_layer, output_layer)

    def create_network(self, hidden_layer, output_layer):
        self.hiddenLayer = []
        self.params = []
        for layer in hidden_layer:
            self.hiddenLayer.append(layer)
            self.params += layer.params
        self.outputLayer = output_layer
        self.output = self.outputLayer.output
        self.params += output_layer.params

    def save(self, filename):
        fptr = open(filename, 'w')
        data = []
        for layer in self.hiddenLayer:
            data.append(layer.get_dict())
        data.append(self.outputLayer.get_dict())
        json.dump(data, fptr)
        fptr.close()

    @staticmethod
    def load(filename, input):
        fptr = open(filename, 'r')
        data = json.load(fptr)
        hidden_layer = []
        for layer_data in data[:-1]:
            layer = Layer.load(layer_data, input=input)
            hidden_layer.append(layer)
            input = layer.output
        output_layer = Layer.load(data[-1], input=hidden_layer[-1].output)
        fptr.close()
        nnet = Network(new_network=False)
        nnet.create_network(hidden_layer, output_layer)
        return nnet


class Layer(object):
    def __init__(self, name, n_in, n_out, input=T.matrix('x'), rng=numpy.random.RandomState(), bias=False,
                 activation="tanh", W=None, b=None):
        self.input = input
        self.name = name
        self.activation = activation

        if W == None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-(numpy.sqrt(6.) / numpy.sqrt(n_in + n_out + 1)),
                    high=(numpy.sqrt(6.) / numpy.sqrt(n_in + n_out + 1)),
                    size=(n_in, n_out)),
                dtype=theano.config.floatX
            )
            if self.get_activation() == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        if b == None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.b = b

        if activation == "none":
            lin_output = input
        else:
            lin_output = T.dot(input, self.W) + self.b

        activation_fn = self.get_activation()
        self.output = (
            lin_output if activation_fn is None
            else activation_fn(lin_output)
        )

        if activation == "none":
            self.params = []
        elif bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

    def get_activation(self):
        if self.activation == "tanh":
            return T.tanh
        elif self.activation == "sigmoid":
            return T.nnet.sigmoid
        elif self.activation == "relu":
            return T.nnet.relu
        elif self.activation == "linear":
            return None
        elif self.activation == "none":
            return None

    def get_dict(self):
        return {'name': self.name,
                'W': self.W.get_value().tolist(),
                'b': self.b.get_value().tolist(),
                'activation': self.activation,
                'in': self.W.get_value().shape[0],
                'out': self.b.get_value().shape[0],
                'bias': True if len(self.params) > 1 else False}

    @staticmethod
    def load(layer_data, input=T.matrix('x')):
        W = theano.shared(value=numpy.asarray(layer_data['W'], dtype=theano.config.floatX), name='W', borrow=True)
        b = theano.shared(value=numpy.asarray(layer_data['b'], dtype=theano.config.floatX), name='b', borrow=True)
        return Layer(name=layer_data['name'],
                     input=input,
                     W=W,
                     b=b,
                     activation=layer_data['activation'],
                     n_in=layer_data['in'],
                     n_out=layer_data['out'],
                     bias=layer_data['bias'])
