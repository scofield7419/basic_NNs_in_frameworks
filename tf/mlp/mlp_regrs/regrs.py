# coding=utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_size = 0
batch_size = 50
keep_prob = 1
features_num = 6


def read_data():
    global data_size
    data = pd.read_csv('bj_housing.csv')
    # data = pd.read_csv('housing.csv')
    length = len(data)
    # print(length)
    # print(int(3 * length / 4))
    X = []
    y = []

    for dx, values in data.iterrows():
        # print(dx,)
        vect = []
        Area = values[0]
        vect.append(Area)
        Living = values[2]
        vect.append(Living)
        School = values[3]
        vect.append(School)
        Year = values[4]
        vect.append(Year)
        Floor = values[5]
        vect.append(Floor)
        Room = values[6]
        vect.append(Room)
        vect = np.array(vect)
        X.append(vect)

        Value = values[1]
        y.append([Value])

        # vect = []
        # Area = values[0]
        # vect.append(Area)
        # Living = values[1]
        # vect.append(Living)
        # School = values[2]
        # vect.append(School)
        # vect = np.array(vect)
        # X.append(vect)
        #
        # Value = values[3]
        # y.append([Value])

    X = np.array(X)
    y = np.array(y)

    data_size = int(3 * length / 4)
    X_train = X[:int(3 * length / 4)]
    y_train = y[:int(3 * length / 4)]
    X_test = X[int(3 * length / 4):]
    y_test = y[int(3 * length / 4):]

    # y_train = train_data['Value']
    # X_train = train_data.drop('Value', axis=1)
    # y_test = train_data['Value']
    # X_test = test_data.drop('Value', axis=1)
    return X_train, y_train, X_test, y_test


def add_layer(input, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + biases  # not actived yet
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=keep_prob_s)
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output


# generate datas
# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise
# print(len(x_data))
# print(x_data)
# print(y_data)

X_train, y_train, X_test, y_test = read_data()
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

# drawing
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# line = ax.plot(range(batch_size), y_train[:batch_size], 'b')
ax.set_ylim([0, 1500])
# ax.set_ylim([200000, 1200000])
plt.ion()
plt.show()

# define layers
xs = tf.placeholder(tf.float32, [None, features_num])
ys = tf.placeholder(tf.float32, [None, 1])
keep_prob_s = tf.placeholder(tf.float32)
layer1 = add_layer(xs, features_num, 10, activation_function=tf.nn.relu)
layer2 = add_layer(layer1, 10, 30, activation_function=tf.nn.relu)
layer3 = add_layer(layer2, 30, 40, activation_function=tf.nn.relu)
layer4 = add_layer(layer3, 40, 20, activation_function=tf.nn.relu)
layer5 = add_layer(layer4, 20, 10, activation_function=tf.nn.relu)
prediction = add_layer(layer5, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for epoch in range(8000):

        # batch
        start = (epoch * batch_size) % data_size
        end = min(start + batch_size, data_size)
        # print(np.shape(X_train[start:end]))
        # print(np.shape(y_train[start:end]))
        _, pred, los = sess.run([train_step, prediction, loss],
                                feed_dict={xs: X_train[start:end], ys: y_train[start:end]
                                    , keep_prob_s: keep_prob})
        # don't need run in batch_size way
        if epoch % 50 == 0:
            print("epoch: ", '%04d' % (epoch + 1), "loss: ", los)
            # print("real values: \n", y_train[start:end])
            # print("pred: \n", pred)

            try:
                ax.lines.remove(line[0])
                ax.lines.remove(lines2[0])
            except:
                pass
            leng = end - start
            line = ax.plot(range(leng), y_train[start:end], 'b')
            lines2 = ax.plot(range(leng), pred[:batch_size], 'r--')
            plt.pause(1)
