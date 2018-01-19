import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils.data as data
from sklearn.feature_extraction import DictVectorizer

epoch_num = 1000
batch_size = 32
batch_num = 0
input_dim = 0

def read_data():
    global data_size, features_num, outputs_num
    data_train = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    selected_teatures = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
    X_train = data_train[selected_teatures]
    y_train = data_train['Survived']
    X_test = test_data[selected_teatures]
    y_test = None

    X_train['Embarked'].fillna('S', inplace=True)
    X_test['Embarked'].fillna('S', inplace=True)
    X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
    X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
    X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
    # print(X_train)
    # vectorize features
    dict_vec = DictVectorizer(sparse=False)
    X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = dict_vec.fit_transform(X_test.to_dict(orient='record'))

    y_train = pd.DataFrame(y_train)
    dict_vec_y = DictVectorizer(sparse=False)
    y_train = dict_vec_y.fit_transform(y_train.to_dict(orient='record'))
    # print(y_train)

    # print(np.shape(X_train))
    data_size = np.shape(X_train)[0]
    features_num = np.shape(X_train)[1]
    outputs_num = np.shape(y_train)[1]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = read_data()

batch_num = len(X_train) // batch_size
print(X_train[0])
print(y_train[:3])
input_dim = len(X_train[0])

# x, y = Variable(X_train), Variable(Y_train)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# class MyDataset(data.Dataset):
#     def __init__(self, images, labels):
#         self.images = images
#         self.labels = labels
#
#     def __getitem__(self, index):#返回的是tensor
#         img, target = self.images[index], self.labels[index]
#         return img, target
#
#     def __len__(self):
#         return len(self.images)
# dataset = MyDataset(images, labels)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(Net, self).__init__()
        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden_1)  # hidden layer
        self.hidden_2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.hidden_3 = torch.nn.Linear(n_hidden_2, n_hidden_3)  # hidden layer
        self.out = torch.nn.Linear(n_hidden_3, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden_1(x))  # activation function for hidden layer
        x = F.relu(self.hidden_2(x))  # activation function for hidden layer
        x = F.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.out(x)
        return x


net = Net(n_feature=input_dim, n_hidden_1=10, n_hidden_2=20, n_hidden_3=10, n_output=1)  # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

for t in range(epoch_num):
    start_i = 0
    for i in range(0, batch_num):
        inputs = X_train[start_i * batch_size:(start_i + 1) * batch_size]
        classes = y_train[start_i * batch_size:(start_i + 1) * batch_size]
        inputs, classes = Variable(torch.Tensor(inputs)), Variable(torch.Tensor(classes))
        out = net(inputs)  # input x and predict based on x
        loss = loss_func(out, classes)  # must be (1. nn output, 2. target), the target label is NOT one-hotted

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if t % 10 == 0:
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = classes.data.numpy()
            accuracy = sum(pred_y == target_y) / 100.
            print('epoch=%4d, Accuracy=%.2f' % (t, accuracy))
        start_i += 1
