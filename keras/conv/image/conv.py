# -*- coding: utf-8 -*-

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization  as bn
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.datasets import mnist
# 调用后端接口
from keras import backend as K
import keras.utils

'''
对于图像的卷积，就是二维卷积，因为图像保持2维shape不变
'''

# 准备数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 转换数据维度
# reshape成(sample_volume, a_vector)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# 转换数据标签
Y_train = (np.arange(10) == y_train[:, None]).astype(int)
Y_test = (np.arange(10) == y_test[:, None]).astype(int)
'''
在预处理的时候把输出，即类别标签已转换为了0，1序列，所以输出不再需要处理。 不过keras自带工具，keras.utils. np_utils
可以完成转换，例如，若y_test为整型的类别标签，Y_test = np_utils.to_categorical(y_test, nb_classes)， Y_test将得到0,1序列化的结果。
第二种方法：
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''
'''
特别注意，对于图像数据，tf与th的处理是不一样的。所以需要判断&转换
'''
# tf或th为后端，采取不同参数顺序
if K.image_data_format() == 'channels_first':
    # -x_train.shape[0]=6000
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    # -x_train.shape:(60000, 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    # x_test.shape:(10000, 1, 28, 28)
    # 单通道灰度图像,channel=1
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

inputs = Input(shape=input_shape)

# 加一层卷积
x = Conv2D(32, kernel_size=(3, 3),
           activation='relu')(inputs)
x = bn()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = bn()(x)

# 加一层卷积
x = Conv2D(64, (3, 3), activation='relu')(x)
x = bn()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = bn()(x)

# 加一层卷积
x = Conv2D(128, (3, 3), activation='relu')(x)
x = bn()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = bn()(x)

# 加Flatten，数据一维化
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = bn()(x)

x = Dense(10, activation='softmax')(x)

model = Model(input=inputs, output=x)

# 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 训练模型，载入数据，verbose=1为输出进度条记录
# validation_data为验证集
model.fit(X_train, Y_train,
          batch_size=128,
          epochs=20,
          shuffle=True, verbose=1, validation_split=0.3)

# 开始评估模型效果
# verbose=0为不输出日志信息
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss: %f, accu: %f' % (score[0], score[1]))
