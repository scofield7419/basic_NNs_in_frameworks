# coding=utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

data_size = 0
batch_size = 20
keep_prob = 0.8
features_num = 6
outputs_num = 1


def read_data():
    global data_size, features_num, outputs_num
    data_train = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    selected_teatures = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
    X_train = data_train[selected_teatures]
    y_train = data_train['Survived']
    X_test = test_data[selected_teatures]
    y_test = None

    X_train['Embarked'].fillna('S', inplace=True)
    X_test['Embarked'].fillna('S', inplace=True)
    X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
    X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
    X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
    # print(X_train)
    # vectorize features
    dict_vec = DictVectorizer(sparse=False)
    X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = dict_vec.fit_transform(X_test.to_dict(orient='record'))

    y_train = pd.DataFrame(y_train)
    dict_vec_y = DictVectorizer(sparse=False)
    y_train = dict_vec_y.fit_transform(y_train.to_dict(orient='record'))
    # print(y_train)

    # print(np.shape(X_train))
    data_size = np.shape(X_train)[0]
    features_num = np.shape(X_train)[1]
    outputs_num = np.shape(y_train)[1]

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


def compute_accuracy(v_xs, v_ys):
    global prediction, keep_prob
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob_s: keep_prob})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob_s: keep_prob})
    return result


X_train, y_train, X_test, y_test = read_data()
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

# define layers
xs = tf.placeholder(tf.float32, [None, features_num])
ys = tf.placeholder(tf.float32, [None, outputs_num])
keep_prob_s = tf.placeholder(tf.float32)
layer1 = add_layer(xs, features_num, 10, activation_function=tf.nn.relu)
layer2 = add_layer(layer1, 10, 20, activation_function=tf.nn.relu)
# layer3 = add_layer(layer2, 20, 30, activation_function=tf.nn.relu)
# layer4 = add_layer(layer3, 30, 10, activation_function=tf.nn.relu)
prediction = add_layer(layer2, 20, outputs_num, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
# 基于min和max对张量t进行截断操作，为了应对梯度爆发或者梯度消失的情况
# cross_entropy = -tf.reduce_mean(
#     tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)), reduction_indices=[1]))
# cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(ys, prediction))
train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for epoch in range(1000):
        # batch
        start = (epoch * batch_size) % data_size
        end = min(start + batch_size, data_size)
        _, pred, los = sess.run([train_step, prediction, cross_entropy],
                                feed_dict={xs: X_train[start:end], ys: y_train[start:end]
                                    , keep_prob_s: keep_prob})
        # don't need run in batch_size way
        if epoch % 50 == 0:
            print("epoch: ", '%04d' % (epoch + 1), "loss: ", los, "accuracy: ",

                  compute_accuracy(X_train[start:end], y_train[start:end]))
            # print("real values: \n", y_train[start:end])
            # print("pred: \n", pred)
