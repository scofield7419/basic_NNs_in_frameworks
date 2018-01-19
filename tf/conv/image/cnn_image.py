# coding=utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# -------------------------------------------
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)


# -------------------------------------------
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# -------------------------------------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# -------------------------------------------
def conv2d(x, W):
    ####!!!!注意：因为是图像数据，所以用conv2d二维的卷积。相应的输入tensor是4维的！
    #### 如果是文本数据，那么输入应该3维的，并且应该用Conv1d卷积。

    # W, filter, shape:[filter_height, filter_width, in_channels, out_channels]
    # 注意， filter的height, width决定卷积效果，不影响output的shape

    # x, input, shape:[batch, in_height, in_width, in_channels]

    # stride [1, x_movement, y_movement, 1], 每次卷积以后卷积窗口在input中滑动的距离
    # Must have strides[0] = strides[3] = 1

    # padding ：有SAME和VALID两种选项，表示是否要保留图像边上那一圈不完全卷积的部分。如果是SAME，则保留

    # output shape: [filter_height/strde_x_movement, filter_width/strde_y_movement, out_channels]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1], input, strides, padding 与conv2d一样
    # ksize: 长为4的list,表示池化窗口的尺寸,
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# -------------------------------------------
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1],shape:[batch, in_height, in_width, in_channels]

# -------------------------------------------
## conv1 layer ##
# filter,shape:[filter_height, filter_width, in_channels, out_channels]
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])

# output shape: [filter_height/conv2d_strde_x_movement, conv2d_filter_width/strde_y_movement, out_channels]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32

# output shape: [input_height/pool_strde_x_movement, input_width/pool_strde_y_movement, out_channels]
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

# -------------------------------------------
## conv2 layer ##
# input size 14x14x32
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])

# output shape: [filter_height/conv2d_strde_x_movement, conv2d_filter_width/strde_y_movement, out_channels]
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64

# output shape: [input_height/pool_strde_x_movement, input_width/pool_strde_y_movement, out_channels]
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

# -------------------------------------------
## fc1 layer ##
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64],展平开
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# -------------------------------------------
## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# -------------------------------------------
# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# -------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, loss = sess.run([train_step, cross_entropy], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print("epoch: ", i + 1, "loss: ", loss,
                  "accu: ", compute_accuracy(mnist.test.images, mnist.test.labels))

# -------------------------------------------




'''
```````api学习``````
一、tf.truncated_normal:
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
shape: 一维的张量，也是输出的张量。
mean: 正态分布的均值。
stddev: 正态分布的标准差。
dtype: 输出的类型。
seed: 一个整数，当设置之后，每次生成的随机数都一样。
name: 操作的名字。

从截断的正态分布中输出随机值。生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。

在正态分布的曲线中，横轴区间（μ-σ，μ+σ）内的面积为68.268949%。 
横轴区间（μ-2σ，μ+2σ）内的面积为95.449974%。 
横轴区间（μ-3σ，μ+3σ）内的面积为99.730020%。 
X落在（μ-3σ，μ+3σ）以外的概率小于千分之三，在实际问题中常认为相应的事件是不会发生的，基本上可以把区间（μ-3σ，μ+3σ）看作是随机变量X实际可能的取值区间，这称之为正态分布的“3σ”原则。 
在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。

a = tf.Variable(tf.random_normal([2,2],seed=1))
print(sess.run(a))
    
输出：
[[-0.81131822  1.48459876]
 [ 0.06532937 -2.44270396]]

二、tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
从正态分布中输出随机值。 

shape: 一维的张量，也是输出的张量。
mean: 正态分布的均值。
stddev: 正态分布的标准差。
dtype: 输出的类型。
seed: 一个整数，当设置之后，每次生成的随机数都一样。
name: 操作的名字。


b = tf.Variable(tf.truncated_normal([2,2],seed=2))
print(sess.run(b))


[[-0.85811085 -0.19662298]
 [ 0.13895047 -1.22127688]]
 
'''
