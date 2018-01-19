# -*- coding: utf-8 -*-

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization  as bn
import numpy as np
import keras.utils
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing import sequence

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3  # 3-gram
hidden_dims = 250
epochs = 5

# 准备数据
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences:(samples, time)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()

# 加一层embedding
'''
input_dim:大或等于0的整数，字典长度，即输入数据最大下标+1 
output_dim:大于0的整数，代表全连接嵌入的维度
input_length:当输入序列的长度固定时，该值为其长度。如果要在该层后接 Flatten 层，
    然后 接 Dense 层，则必须指定该参数，否则 Dense 层的输出维度无法自动推断。
'''
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))

'''
!!!!!!!重点！！！！！！
SimpleRNN/LSTM/GRU 三个叠层起来要能跑，必须：
return_sequences=True ；即让他们输出times_step的seq结构的结果让下一步的SimpleRNN/LSTM/GRU使用

对于普通Dense,则需要使用TImeDistributed包装起来，为一个time_step接一个全连接，输出seq结构的值给下一步使用。

理解：
1、没有seq结构的结果，为_output张量值作为结果
2、seq结构的结果，则会输出计算每个time_step状态的张量值作为结果
****所以，一般使用time_step值得场景是做ner, labeling, seq2seq下
'''

'''
output_dim: Positive integer, dimensionality of the output space.
'''
model.add(SimpleRNN(128, dropout=0.2, activation='relu',return_sequences=True ,kernel_initializer='glorot_uniform'))
model.add(LSTM(128, dropout=0.2, activation='relu', init='glorot_uniform',return_sequences=True))
model.add(GRU(128, dropout=0.2, activation='tanh', init='glorot_uniform'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True, verbose=1, validation_split=0.3)
print('test set')
loss, accu = model.evaluate(x_test, y_test, batch_size=100, verbose=1)
print(loss, accu)
