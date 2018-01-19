# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np

'''
keras写一个层，是不用考虑矩阵的尺寸对接，直接指定某隐层节点数，自动计算出入数据维度（即变量矩阵的维度），这点与tf相反。
'''

np.random.seed(1)

model = Sequential()

# Dense是全连接网络,只有第一层需要设定输入层的结点个数,其他都不需要
'''
下面的三个指定输入数据shape的方法是严格等价的:
    model.add(Dense(32, input_shape=(784,)))
    model.add(Dense(32, batch_input_shape=(None, 784)))
        # note that batch dimension is "None" here,
        # so the model will be able to process batches of any size.</pre>
    model.add(Dense(32, input_dim=784))
下面三种方法也是严格等价的:
    model.add(LSTM(32, input_shape=(10, 64)))
    model.add(LSTM(32, batch_input_shape=(None, 10, 64)))
    model.add(LSTM(32, input_length=10, input_dim=64))
'''
model.add(Dense(500, kernel_initializer='glorot_uniform', input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(500, kernel_initializer='glorot_uniform', ))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 输出结果是10个类别，所以维度是10
model.add(Dense(10, kernel_initializer='glorot_uniform', ))
# 最后一层用softmax
model.add(Activation('softmax'))

model.summary()
# 设定训练参数
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# 使用交叉熵损失函数
'''
二分类：binary_crossentropy
多分类：categorical_crossentropy
回归：mse
'''
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# 准备数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(y_train[:3]) #[5 0 4]
# 转换数据维度
# reshape成(sample_volume, a_vector)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# 转换数据标签
Y_train = (np.arange(10) == y_train[:, None]).astype(int)
Y_test = (np.arange(10) == y_test[:, None]).astype(int)

print(Y_train.shape)
print(X_train.shape)
print(X_train[:2])
print(Y_train[:2])
print(Y_test.shape)
print(X_test.shape)
'''
(60000, 10)
(10000, 10)
[0 0 0 0 0 1 0 0 0 0]
(784,)
'''

'''
# 训练过程
for step in range(501):
    # 进行训练, 返回损失(代价)函数
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print 'loss: ', cost
'''

'''
Keras以Numpy数组作为输入数据和标签的数据类型
'''
model.fit(X_train, Y_train, batch_size=128, epochs=10, shuffle=True, verbose=1, validation_split=0.3)
print('test set')
loss, accu = model.evaluate(X_test, Y_test, batch_size=100, verbose=1)
print(loss, accu)
