# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import pandas

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
print(Y_train[:2])
print(X_train.shape)
print(Y_train.shape)
print()
print(X_test.shape)
print(Y_test.shape)
'''
(7999, 6)
(7999,)
(7999, 6)
(7999,)
'''

np.random.seed(7)

model = Sequential()
model.add(Dense(20, input_shape=(6,), init='uniform', activation='relu'))
model.add(Dense(40, init='uniform', activation='relu'))
model.add(Dense(60, init='uniform', activation='relu'))
model.add(Dense(40, init='uniform', activation='relu'))
model.add(Dense(20, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='linear'))  # 也可以不加activation项

# 使用交叉熵损失函数
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])

model.fit(X_train, Y_train, batch_size=32, epochs=20, shuffle=True, verbose=1, validation_split=0.3)
print('\ntest set:')
loss = model.evaluate(X_test, Y_test, batch_size=20, verbose=1)
print("loss: %d, mean_absolute_error: %f, acc: %f" % (loss[0], loss[1], loss[2]))
