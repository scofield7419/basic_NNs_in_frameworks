# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import pandas
from keras.layers.normalization import BatchNormalization  as bn
from keras.layers import Input, Dense
from keras.models import Model

'''
1. 使用models接口
2. 使用keras 泛型编程API
'''

# load dataset
dataframe = pandas.read_csv("bj_housing.csv", header=0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
train = dataset[:len(dataset) * 9 // 10]
X_train = np.array([[row[i] for i in range(0, len(row)) if i != 1] for row in train])
Y_train = np.array([[x[1]] for x in train])  # train[:,:1]
test = dataset[len(dataset) * 9 // 10:]
X_test = np.array([[row[i] for i in range(0, len(row)) if i != 1] for row in train])
Y_test = np.array([[x[1]] for x in train])  # train[:,:1]

'''
泛型：(x)
bn():BatchNormalization
'''
inputs = Input(shape=(6,))
x = Dense(20)(inputs)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(40)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(60)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(40)(x)
x = bn()(x)
x = Activation('relu')(x)
x = Dense(20)(x)
x = bn()(x)
x = Activation('relu')(x)
predictions = Dense(1, activation='linear')(x)

model = Model(input=inputs, output=predictions)
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mae', 'acc'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
          nb_epoch=20, batch_size=32)
print('\ntest set:')
loss = model.evaluate(X_test, Y_test, batch_size=20, verbose=1)
print("loss: %d, mean_absolute_error: %f, acc: %f" % (loss[0], loss[1], loss[2]))
