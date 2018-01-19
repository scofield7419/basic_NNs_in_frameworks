# coding=utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# -------------------------------------------
# 写RNN与MLP完全不一样的思路。
# 1. RNN是需要构建出基本cell，然后指派给dynamic_rnn形成一层确定的RNN
# 2. RNN结构，在构造cell时，依然要将ax+b的结果传入cell中。而激活函数就被cell接管了。
#   特别地，对于RNNcell.很简单，即：tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
# -------------------------------------------
lr = 0.001
epoch = 1000
keep_prob = 1

time_step = 28
batch_size = 128
input_size = 28
hidden_units = 50
output_size = 10

layer_num = 3

# -------------------------------------------
# this is data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# trX = trX.reshape(-1, 28, 28)
# teX = teX.reshape(-1, 28, 28)
# print(tf.__version__)

step_size_in_epoch = mnist.train.num_examples // batch_size
# print(np.shape(trX))
# print(step_size_in_epoch)
# -------------------------------------------

# 一开始是输入数据：
# RNN 的输入shape = (batch_size, timestep_size, input_size)
# 1. basic_cell自动实现input_size到hidden_unit的矩阵变化。
# 2. timestep_size一般设定为一个序列的长度。如果是一张图片的话，那么将行列拆分为timestep_size x input_size的格式
# X = tf.reshape(_X, [-1, 28, 28])

# tf Graph input
x = tf.placeholder(tf.float32, [None, time_step, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

# -------------------------------------------

# cell有这些类型：BasicRNNCell/BasicLSTMCell/////GRUCell/RNNCell/LSTMCell
# basic 与后面的区别在于：后面的定制度高

rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, forget_bias=1.0, state_is_tuple=True)
# -------------------------------------------

# 可考虑dropout选项，在cell外部包裹上dropout，这个类叫DropoutWrapper

rnn_cell = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
# -------------------------------------------

# 如果是要构建多层的RNN，可以：MultiRNNCell

# mrnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell for _ in range(layer_num)], state_is_tuple=True)
# -------------------------------------------

# 注意到RNN都是有中间的state输出的，所以用一开始全零来初始化state，后面每一次batch都在上一batch的state值下更新
# init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

init_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
# -------------------------------------------

# 构建完整一栋RNN的API有这些：dynamic_rnn/static_rnn/state_saving_rnn/bidirectional_rnn
# 在上述api中，要传入initial_state:state
#  这是因为每训练一次batch会得到一个final_state,在进入下个batch的训练时，需要将final_state赋值给init_state。

rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=init_state)
# -------------------------------------------
# 取最后一个时序的output,
# 展开output,再矩阵变换形状
# shape = (batch * steps, cell_size)
rnn_output = rnn_outputs[:, -1, :]
l_out_x = tf.reshape(rnn_output, [-1, hidden_units])

weights = tf.Variable(tf.random_normal([hidden_units, output_size]))
bias = tf.Variable(tf.constant(0.1, shape=[output_size, ]))

results = tf.matmul(l_out_x, weights) + bias
# -------------------------------------------
# calcu loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=results, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

# -------------------------------------------
# calcu accuracy
correct_pred = tf.equal(tf.argmax(tf.nn.softmax(results), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# -------------------------------------------

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(epoch):
        # train_loss = 0
        # 保存每次执行后的最后状态，然后赋给下一次执行
        # training_state = tf.zeros((batch_size, hidden_units))

        # init_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        for step in range(step_size_in_epoch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # batch_xs ==> (128 batch, 28 steps, 128 hidden)
            batch_xs = batch_xs.reshape(batch_size, time_step, input_size)
            # 注意不能直接计算，需要让sess在图中计算！！！
            # batch_xs = tf.convert_to_tensor(batch_xs, dtype=tf.float32)
            # batch_xs, batch_ys = sess.run([batch_xs, batch_ys])
            # 注意，结果名称不可与计算图的变量名重名，如果存储结果的变量和tensorflow的节点重名！！！会报错！！
            _, tr_loss, tr_accuracy = sess.run(
                [train_op, loss, accuracy],
                feed_dict={
                    y: batch_ys,
                    x: batch_xs,
                })
            # train_loss += tr_loss

            if (step + 1) % step_size_in_epoch == 0:
                print("epoch: ", '%04d' % (ep + 1), "loss: ", tr_loss, "accuracy: ", tr_accuracy)
# -------------------------------------------




# -------------------------------------------





# ```````api学习``````
# 一、tf.transpose:
# original: perm = [2,1,0], after:[0, 2, 1]
# [[[1 2 3]
#   [4 5 6]]
#  [[7 8 9]
#   [10 11 12]]]
# ->
# [[[1 4]
#   [2 5]
#   [3 6]]
#  [[7 10]
#   [8 11]
#   [9 12]]]
#
#
# 二、tf.unpack/tf.pack(tf1.0+已经将pack函数更名为stack函数,unpack函数对应更名为unstack函数.)
# 根据axis分解成num个张量，返回的值是list类型，如果没有指定num则根据axis推断出
# stack -> [array1, array2,...]
# unstack -> [[ ], [ ],...]
