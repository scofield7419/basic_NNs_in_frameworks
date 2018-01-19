import numpy as np
from sklearn.metrics import accuracy_score
import torch as pt
import torch.utils.data as data
import io
import os
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import sys
import torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# current_dir = os.path.dirname(os.path.abspath(__file__))
# top_dir = os.path.dirname(current_dir)
# data_dir = os.path.join(top_dir, 'data')
# sys.path.append(top_dir)
data_dir = r'/Users/scofield/workplaces/pythons/deep_networks/scott_trials/basic_trials/pytorch/conv/images/data'

def accuracy_calculate(pred, true):
    accuracy = accuracy_score(true, pred)
    return accuracy


def _calculate_conv_size(pre_size, conv_kernel_size, stride_step=(1, 1)):
    '''stride step is set to (1,1)'''
    return (pre_size[0] - conv_kernel_size + 1, pre_size[1] - conv_kernel_size + 1)


def _pooling(conv_size, spatial_extent, stride):
    # TODO, the result might be float
    width = (conv_size[0] - spatial_extent) / stride + 1
    height = (conv_size[1] - spatial_extent) / stride + 1
    if (not width.is_integer()) or (not height.is_integer()):
        print("width: {}, height: {}".format(width, height))
        raise Exception("please redesign the F or S of pooling")
    else:
        return (width, height)


# image_size = (100, 1, 28, 28)
# conv1_size = _calculate_conv_size(image_size, 5)
# conv1_size = _pooling(conv1_size, 2, 2)
# conv2_size = _calculate_conv_size(conv1_size, 5)
# conv2_size = _pooling(conv2_size, 2, 2)

'''
movan style
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization
'''

class CNN(nn.Module):
    def __init__(self, image_size):
        super(CNN, self).__init__()

        # image-2d-size
        image_2d_size = tuple(image_size[2:4])

        # set channel
        input_img_channel_num = int(image_size[1])
        conv1_channel_num = 6
        conv2_channel_num = 16

        # set kernel size
        conv1_kernel_size = 5
        conv2_kernel_size = 5

        # set pooling stride, spatial_extent
        F = 2
        S = 2

        # calculate the flatten size for the conv output
        conv1_size = _calculate_conv_size(image_2d_size, conv1_kernel_size)
        conv1_size = _pooling(conv1_size, F, S)
        conv2_size = _calculate_conv_size(conv1_size, conv2_kernel_size)
        conv2_size = _pooling(conv2_size, F, S)
        conv_output_flatten_size = int(conv2_channel_num * conv2_size[0] * conv2_size[0])

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(input_img_channel_num, conv1_channel_num, conv1_kernel_size)
        self.conv2 = nn.Conv2d(conv1_channel_num, conv2_channel_num, conv2_kernel_size)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(conv_output_flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# config
BATCH_SIZE = 100
MAX_TRAINING_EPOCH = 30
#

# read dataset
training_data_path = os.path.join(data_dir, 'mnist_training_set')
test_data_path = os.path.join(data_dir, 'mnist_test_set')
training_data = pickle.load(open(training_data_path, 'rb'))
test_data = pickle.load(open(test_data_path, 'rb'))

# read into tensor
training_data = pt.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE)
test_data = pt.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
#

# build cnn
img_size = (1, 1, 28, 28)  # set the image size manually
cnn1 = CNN(img_size)
print(cnn1)
#

# loss func and optim
optimizer = pt.optim.SGD(cnn1.parameters(), lr=0.01, momentum=0.9)
lossfunc = pt.nn.CrossEntropyLoss()
#

# # TEST CNN
# input = Variable(pt.randn(1, 1, 28, 28)).cuda()
# out = cnn1(input)
# print(out)
# sys.exit()
# #

# ----------------------------------------------------------------------------------------------------------------------
# training
for epoch in range(MAX_TRAINING_EPOCH):

    for i, data in enumerate(training_data):

        optimizer.zero_grad()

        (inputs, true_labels) = data

        inputs = pt.autograd.Variable(inputs)
        true_labels = pt.autograd.Variable(true_labels)

        outputs = cnn1(inputs)  # -> return pt.nn.functional.softmax(self.fc3(dout))

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
    outputs = cnn1(inputs)

    outputs = outputs.cpu().data.numpy()
    true_labels = true_labels.cpu().data.numpy()
    pred_labels = [np.argmax(x) for x in outputs]
    accuracy = accuracy_calculate(pred_labels, true_labels)
    accuracy_list.append(accuracy)

print("test avg_accuracy: ", sum(accuracy_list) / len(accuracy_list))
# ----------------------------------------------------------------------------------------------------------------------
