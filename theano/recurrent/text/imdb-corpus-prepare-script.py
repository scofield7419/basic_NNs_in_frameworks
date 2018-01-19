import pickle, gzip
import numpy as np
import theano

# dic info
with gzip.open('data/imdb.dict.pkl.gz', 'rb') as f:
    word2idx = pickle.load(f, encoding='bytes')
word2idx = {w.decode(): idx for w, idx in word2idx.items()}
word2idx['#UNDEF'] = 0
word2idx['#UNDEF2'] = 1

pat2word = None

# corpus info
maxlen = 200
with open('data/imdb.pkl', 'rb') as f:
    train_x, train_y = pickle.load(f, encoding='bytes')
    test_x, test_y = pickle.load(f, encoding='bytes')
data_x = train_x + test_x
data_y = train_y + test_y

data_x_len = np.array([len(x) for x in data_x])
data_y = np.array(data_y, dtype='int32')[data_x_len <= maxlen]
data_x = [x for x, l in zip(data_x, data_x_len) if l <= maxlen]

n_data = len(data_x)
# 0 is #UNDEF by default
data_x_matrix = np.zeros((n_data, maxlen), dtype='int32')
data_mask = np.zeros((n_data, maxlen), dtype=theano.config.floatX)
for idx, x in enumerate(data_x):
    data_x_matrix[idx, :len(x)] = x
    data_mask[idx, :len(x)] = 1.

# shuffle the samples
idx_seq = np.arange(n_data)
np.random.shuffle(idx_seq)
data_x_matrix = data_x_matrix[idx_seq]
data_mask = data_mask[idx_seq]
data_y = data_y[idx_seq]

with open('data/imdb-prepared.pkl', 'wb') as f:
    pickle.dump(word2idx, f)
    pickle.dump(pat2word, f)
    pickle.dump(data_x_matrix, f)
    pickle.dump(data_mask, f)
    pickle.dump(data_y, f)
