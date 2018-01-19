import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

'''
pytorch是一个动态的建图的工具。不像Tensorflow那样，先建图，然后通过feed和run重复执行建好的图。
相对来说，pytorch具有更好的灵活性。

pytorch中有两种变量类型，一个是Tensor，一个是Variable。

'''
#####generate data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(1000, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
print(x)
'''
直接基于输入数据产生变量，可训练
因为pytorch是动态图，所以变量直接输入到定义的net函数中,无需在train中统一考虑传入。
'''
x, y = Variable(x), Variable(y)

'''
继承torch.nn.Module
这里就是定义一层网络
'''
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        '''
        torch.nn.Linear相当于是做一个w*x + b的操作
        '''
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)  # define one_layer network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()  # something about plotting

for t in range(1000):
    prediction = net(x)  # input x and predict based on x

    loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 5 == 0:
        # plot and show learning process: loss
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(-0.5, 0, 'Loss=%.4f, epoch=%4d' % (loss.data[0], t), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
