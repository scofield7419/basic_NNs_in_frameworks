import numpy as np
from sklearn.metrics import accuracy_score
import torch as pt
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import torch.utils.data as data
from sklearn.feature_extraction import DictVectorizer
import io
import os
import pickle
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# current_dir = os.path.dirname(os.path.abspath(__file__))
# top_dir = os.path.dirname(current_dir)
# data_dir = os.path.join(top_dir, 'data')
# sys.path.append(top_dir)
data_dir = r'/Users/scofield/workplaces/pythons/deep_networks/scott_trials/basic_trials/pytorch/mlp/claz/data_loader/data'

def accuracy_calculate(pred, true):
    accuracy = accuracy_score(true, pred)
    return accuracy


class MLP(pt.nn.Module):
    def __init__(self, n_feature, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(MLP, self).__init__()
        self.hidden_1 = pt.nn.Linear(n_feature, n_hidden_1)  # hidden layer
        self.hidden_2 = pt.nn.Linear(n_hidden_1, n_hidden_2)
        self.hidden_3 = pt.nn.Linear(n_hidden_2, n_hidden_3)  # hidden layer
        self.out = pt.nn.Linear(n_hidden_3, n_output)  # output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.hidden_1(x))  # activation function for hidden layer
        x = F.relu(self.hidden_2(x))  # activation function for hidden layer
        x = F.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.out(x)
        return x


# config
BATCH_SIZE = 100
MAX_TRAINING_EPOCH = 30
#

# read dataset
# print(data_dir)
training_data_path = os.path.join(data_dir, 'mnist_training_set')
test_data_path = os.path.join(data_dir, 'mnist_test_set')
training_data = pickle.load(open(training_data_path, 'rb'))
test_data = pickle.load(open(test_data_path, 'rb'))

# read into tensor
training_data = pt.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
test_data = pt.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

mlp1 = MLP(n_feature=784, n_hidden_1=512, n_hidden_2=512, n_hidden_3=128, n_output=10)

# loss func and optim
optimizer = pt.optim.SGD(mlp1.parameters(), lr=0.01, momentum=0.9)
lossfunc = pt.nn.CrossEntropyLoss()

# ----------------------------------------------------------------------------------------------------------------------
# training
for epoch in range(MAX_TRAINING_EPOCH):

    for i, data in enumerate(training_data):

        optimizer.zero_grad()

        (inputs, true_labels) = data

        inputs = pt.autograd.Variable(inputs)
        true_labels = pt.autograd.Variable(true_labels)

        outputs = mlp1(inputs)  # -> return pt.nn.functional.softmax(self.fc3(dout))

        # print("inputs: ", inputs)
        # sys.exit()
        # # see grads
        # for parameter in mlp1.parameters():
        #     print ("grad: ", parameter.grad)
        # sys.exit()
        # #

        loss = lossfunc(outputs, true_labels)
        loss.backward()  # -> accumulates the gradient (by addition) for each parameter

        optimizer.step()  # -> update weights and biases

        if i % 100 == 0:
            outputs = outputs.cpu().data.numpy()
            true_labels = true_labels.cpu().data.numpy()
            pred_labels = [np.argmax(x) for x in outputs]
            accuracy = accuracy_calculate(pred_labels, true_labels)
            print("epoch: {}, batch: {}, accuracy: {}".format(epoch, i, accuracy))


# ----------------------------------------------------------------------------------------------------------------------
# testing
accuracy_list = []
for i, data in enumerate(test_data):
    (inputs, true_labels) = data
    inputs = pt.autograd.Variable(inputs)
    true_labels = pt.autograd.Variable(true_labels)
    outputs = mlp1(inputs)

    outputs = outputs.cpu().data.numpy()
    true_labels = true_labels.cpu().data.numpy()
    pred_labels = [np.argmax(x) for x in outputs]
    accuracy = accuracy_calculate(pred_labels, true_labels)
    accuracy_list.append(accuracy)

print("avg_accuracy: ", sum(accuracy_list) / len(accuracy_list))


# ----------------------------------------------------------------------------------------------------------------------
# save model or parameters
# ----------------------------------------------------------------------------------------------------------------------
# # only save paramters
# pt.save(model.state_dict(), "path1")
# # save model
# pt.save(model, "path1")
# ----------------------------------------------------------------------------------------------------------------------
