# coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
from datetime import timedelta
from collections import Counter
import tensorflow.contrib.keras as kr
from sklearn import metrics

# -------------------------------------------
# RNN, 考虑结合tf的API加入embedding
# -------------------------------------------
# 模型参数
embedding_dim = 64  # 词向量维度
seq_length = 600  # 序列长度(time_step),基本上长度要涵盖上一篇文档的大小
num_classes = 14  # 类别数
vocab_size = 6000  # 词汇表size

num_layers = 2  # 隐藏层层数
hidden_dim = 128  # 隐藏层神经元

dropout_keep_prob = 0.8  # dropout保留比例
learning_rate = 1e-3  # 学习率

batch_size = 128  # 每批训练大小
num_epochs = 10  # 总迭代轮次
print_per_batch = 100  # 每多少轮输出一次结果

# -------------------------------------------
base_dir = r'//Users/scofield/workplaces/pythons/deep_networks/scott_trials/basic_trials/tf/recurrent/rnn_lstm_text/data/'

save_dir = '/Users/scofield/workplaces/pythons/deep_networks/scott_trials/basic_trials/tf/recurrent/rnn_lstm_text/checkpoint'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

dirname = r'/Users/scofield/workplaces/pythons/deep_networks/scott_trials/basic_trials/tf/recurrent/rnn_lstm_text/raw_datasets'


# -------------------------------------------
# 预处理数据，将原始文件夹与分类信息处理到train,test,val数据集
def prepare_data_once():
    """
    将多个文件整合并存到3个文件中
    dirname: 原数据目录
    文件内容格式:  类别\t内容
    """

    def _read_file(filename):
        """读取一个文件并转换为一行"""
        # print(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')

    f_train = open(base_dir + 'cnews.train.txt', 'w', encoding='utf-8')
    f_test = open(base_dir + 'cnews.test.txt', 'w', encoding='utf-8')
    f_val = open(base_dir + 'cnews.val.txt', 'w', encoding='utf-8')
    for category in os.listdir(dirname):  # 分类目录
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            if cur_file != '.DS_Store':
                filename = os.path.join(cat_dir, cur_file)
                content = _read_file(filename)
                if count < 250:
                    f_train.write(category + '\t' + content + '\n')
                elif count < 500:
                    f_test.write(category + '\t' + content + '\n')
                else:
                    f_val.write(category + '\t' + content + '\n')
                count += 1

        print('Finished:', category)

    f_train.close()
    f_test.close()
    f_val.close()

    print(len(open(base_dir + 'cnews.train.txt', 'r', encoding='utf-8').readlines()))
    print(len(open(base_dir + 'cnews.test.txt', 'r', encoding='utf-8').readlines()))
    print(len(open(base_dir + 'cnews.val.txt', 'r', encoding='utf-8').readlines()))


# -------------------------------------------
# 公共实用函数
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {
            input_x: x_batch,
            input_y: y_batch,
            keep_prob: 1.0
        }
        ev_loss, ev_acc = sess.run([loss, acc], feed_dict=feed_dict)
        total_loss += ev_loss * batch_len
        total_acc += ev_acc * batch_len

    return total_loss / data_len, total_acc / data_len


def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    # max_length为sequence_legth,即time_step
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)

    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad


# -------------------------------------------
# 为文本提供batch迭代器
def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    global print_per_batch
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    print_per_batch = num_batch // 2

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


# -------------------------------------------
# 处理文本的读取、输入,准备数据
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

# 看data文件夹下是否有三项文件，没有，就将原始文件夹与分类信息处理到train,test,val数据集
if not (os.path.exists(train_dir) and os.path.exists(test_dir) and os.path.exists(val_dir)):
    prepare_data_once()

if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

# 读取分类类别，固定
categories = ['体育', '财经', '房产', '家居', '彩票', '星座', "社会", '股票',
              '教育', '科技', '时尚', '时政', '游戏', '娱乐']
cat_to_id = dict(zip(categories, range(len(categories))))

# 读取词汇表
words = open_file(vocab_dir).read().strip().split('\n')
word_to_id = dict(zip(words, range(len(words))))
vocab_size = len(words)

# 载入训练集与验证集
print("Load data...")
start_time = time.time()

# 现在开始，查询词汇表和类别表，将文字（train&val）用id编码表示
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)

print("length of train: %5d, val: %5d, test: %5d" % (len(x_train), len(x_val), len(x_test)))

time_dif = get_time_dif(start_time)
print("Time usage for loading data:", time_dif)

# 配置 Saver
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# -------------------------------------------
# 三个待输入的数据
# X的原始输入的最内部的一个维度是batch_size,另一个维度是sequence_length。
# 而通过embedding_lookup将原始输入映射为batch_size, sequence_length, embedding_dim
input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# -------------------------------------------
'''
embedding思考!!!!!非常重要！！！核心！！！敲黑板！！！：
1. 我们希望词向量可以在train的过程中被改变，更适应我们的训练数据,那就要用Variable来表示词向量矩阵。
所以说，embedding是一个变量啊！！！他是会在训练的过程中被动态改变的！

因此我们要定义一个变量来表示，词向量矩阵的维度应该是 vocab_size * rnn_size。 
即每一行代表一个词，列数就是我们需要自己定义的词向量维度。
定义了词向量矩阵的变量，每次输入的时候，还需要为输入的词找到对应的词向量，这些tensorflow都为我们封装好了：
tf.nn.embedding_lookup这个函数就是用于返回所查找的词向量Tensor的。代码如下:

embedding = tf.Variable(tf.truncated_normal([vocab_size, rnn_size], stddev=0.1), name='embedding')
inputs = tf.nn.embedding_lookup(embedding, self.input_data)
　　
2. 实例：

shape:
self.input_data = tf.placeholder(tf.int32, shape=[batch_size, sequence_length], name='input_data')
embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

# 假设有词向量空间x
　　vx = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]
# 假如每个batch有3个句子，每个句子有两个词(sequence_length)，词的id如下
　　ids = [[0,1],[1,2],[0,2]]
　　inputs = tf.nn.embedding_lookup(vx, ids)
# inputs输出结果如下
　　>>> array([[[ 1., 2., 3.],
　　[ 4., 5., 6.]],
　　[[ 4., 5., 6.],
　　[ 7., 8., 9.]],
　　[[ 1., 2., 3.],
　　[ 7., 8., 9.]]], dtype=float32)
输出结果的shape为(3,2,3)

用上述方式就可以查出来一个batch中每个句子的每个词对应的词向量。
所以我们原始输入的batch中，每个元素是一个sequence，sequence中的元素又是每个词对应的id。
https://www.cnblogs.com/chenergougou/p/6898658.html
https://www.cnblogs.com/hellocwh/p/5626457.html

3. input的sequence每个基本单位可以是word，也可以是char,分别对应word embeddings和char embeddings

4. embeddings可以在一开始接受进来做为变量，也可以一开始没有，创建新变量。

    if embedding_matrix != None:
        self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
    else:
        self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])

'''
# 词向量映射
with tf.device('/cpu:0'):
    embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)


# -------------------------------------------
def lstm_cell():  # lstm核
    return tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)


def dropout():  # 为每一个rnn核后面加一个dropout层
    cell = lstm_cell()
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


# -------------------------------------------
with tf.name_scope("rnn"):
    # 多层rnn网络
    cells = [dropout() for _ in range(num_layers)]
    rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
    last = _outputs[:, -1, :]  # 取最后一个time_step时序输出作为结果

    # 全连接层，后面接dropout以及relu激活,有现成API
    fc = tf.layers.dense(last, hidden_dim, name='fc1')
    fc = tf.nn.dropout(fc, keep_prob)
    fc = tf.nn.relu(fc)

    # 分类器
    logits = tf.layers.dense(fc, num_classes, name='fc2')
    # 预测类别
    y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1)  # 预测结果pred,重要输出
# -------------------------------------------
# 损失函数，交叉熵,optimize
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
loss = tf.reduce_mean(cross_entropy)  # loss,重要输出
# 优化器
optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# -------------------------------------------
# 准确率,accuracy
correct_pred = tf.equal(tf.argmax(input_y, 1), y_pred_cls)
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # acc,重要输出

# -------------------------------------------
# starting traing and validating
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print('Training and evaluating...')

    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    start_time_all = time.time()
    for epoch in range(num_epochs):

        print('Epoch:', epoch + 1)

        # 注意，返回一个迭代器！
        batch_train = batch_iter(x_train, y_train, batch_size)
        for x_batch, y_batch in batch_train:
            start_time = time.time()
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                keep_prob: dropout_keep_prob
            }

            # 每多少轮次输出在训练集和验证集上的性能
            if total_batch % print_per_batch == 0:
                feed_dict[keep_prob] = 1.0
                loss_train, acc_train = sess.run([loss, acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(sess, x_val, y_val)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=sess, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Iter Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            # 运行优化
            sess.run(optim, feed_dict=feed_dict)
            total_batch += 1
    str = "all time consumption fot training: {0}"
    print(str.format(get_time_dif(start_time_all)))
    # -------------------------------------------
    # testing
    print('Testing...')
    start_time = time.time()

    # 评估loss and accuracy.
    loss_test, acc_test = evaluate(sess, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    _y_test_cls = np.argmax(y_test, 1)
    _y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            input_x: x_test[start_id:end_id],
            keep_prob: 1.0
        }
        _y_pred_cls[start_id:end_id] = sess.run(y_pred_cls, feed_dict=feed_dict)

    # 评估Precision, Recall and F1-Score.
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(_y_test_cls, _y_pred_cls, target_names=categories))
    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(_y_test_cls, _y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage for testing:", time_dif)

# ```````api学习``````

'''
1.dict(zip(words))
见aa.py


2.conlections.Counter
Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，
其计数作为value。计数值可以是任意的Interger（包括0和负数）
>>> c = Counter("abcdefgab")
>>> c["a"]
2
>>> c["c"]
1
>>> c["h"]
0

3、tf.get_variable(name, shape=None, dtype=dtypes.float32, initializer=None,
                 regularizer=None, trainable=True, collections=None,
                 caching_device=None, partitioner=None, validate_shape=True,
                 custom_getter=None):
如果在该命名域中之前已经有名字=name的变量，则调用那个变量；
如果没有，则根据输入的参数重新创建一个名字为name的变量。在众多的输入参数中，有几个是我已经比较了解的，下面来一一讲一下

name: 这个不用说了，变量的名字 
shape: 变量的形状，[]表示一个数，[3]表示长为3的向量，[2,3]表示矩阵或者张量(tensor) 
dtype: 变量的数据格式，主要有tf.int32, tf.float32, tf.float64等等 
initializer: 初始化工具，有tf.zero_initializer, tf.ones_initializer, tf.constant_initializer, 
tf.random_uniform_initializer, tf.random_normal_initializer, tf.truncated_normal_initializer等

4. tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引，其他的参数不介绍。

c = np.random.random([10,1])  
b = tf.nn.embedding_lookup(c, [1, 3])  

print sess.run(b)  
print c  

[[ 0.77505197]
 [ 0.20635818]]

[[ 0.23976515]
 [ 0.77505197]
 [ 0.08798201]
 [ 0.20635818]
 [ 0.37183035]
 [ 0.24753178]
 [ 0.17718483]
 [ 0.38533808]
 [ 0.93345168]
 [ 0.02634772]]

'''
