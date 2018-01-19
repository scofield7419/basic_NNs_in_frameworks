# -*- coding: utf-8 -*-

from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional

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
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Bidirectional(LSTM(128, dropout=0.2)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
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
