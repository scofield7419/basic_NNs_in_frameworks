import numpy as np
import theano
import theano.tensor as T


#####################################
# 实用函数
def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction) / len(correct_prediction)
    return accuracy


#####################################
# 实用函数

rng = np.random

N = 400  # training sample size
input_dim = 784  # number of input variables
hiden_dim_1 = 20
hiden_dim_2 = 40
output_dim = 1
# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, input_dim), rng.randint(size=N, low=0, high=2))
print(D[0])
print(D[1])

#####################################
# 输入&变量
# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weights and biases
W_1 = theano.shared(rng.randn(input_dim), name="W_1")
b_1 = theano.shared(0.1, name="b_1")
W_2 = theano.shared(rng.randn(hiden_dim_1), name="W_2")
b_2 = theano.shared(0.1, name="b_2")
W_3 = theano.shared(rng.randn(hiden_dim_2), name="W_3")
b_3 = theano.shared(0.1, name="b_3")

#####################################
# 构建计算图
# Construct Theano expression graph !!!!!!!
p_1 = T.nnet.relu(T.dot(x, W_1) + b_1)  # Logistic Probability that target = 1 (activation function)
p_2 = T.nnet.relu(T.dot(p_1, W_2) + b_2)  # Logistic Probability that target = 1 (activation function)
p_3 = T.nnet.sigmoid(T.dot(p_2, W_3) + b_3)  # Logistic Probability that target = 1 (activation function)

# The prediction thresholded,直接让结果输出为0、1
# prediction = 1 if p_3 > 0.5 else 0
prediction = p_3 > 0.5

# compute the cost
# msq for reg
# cost = T.mean(T.square(l2.outputs - y))
# cross_entropy for clz
xent = (T.nnet.binary_crossentropy(prediction, y)).mean()
cost = xent + 0.01 * (W_1 ** 2).sum() + 0.01 * (W_2 ** 2).sum() + 0.01 * (W_3 ** 2).sum()

# compute the gradients
gW1, gb1, gW2, gb2, gW3, gb3 = T.grad(cost, [W_1, b_1, W_2, b_2, W_3, b_3])

# Compile
learning_rate = 0.1
train = theano.function(
    inputs=[x, y],
    outputs=[prediction, xent],  # cost 加入正则项后，不能直接作为优化对象。
    updates=[
        (W_1, W_1 - learning_rate * gW1), (b_1, b_1 - learning_rate * gb1),
        (W_2, W_2 - learning_rate * gW2), (b_2, b_2 - learning_rate * gb2),
        (W_3, W_3 - learning_rate * gW3), (b_3, b_3 - learning_rate * gb3)])

predict = theano.function(inputs=[x], outputs=prediction)

#####################################
# Training
for i in range(1000):
    _, loss = train(D[0], D[1])
    if i % 50 == 0:
        print('cost:', loss)
        print("accuracy:", compute_accuracy(D[1], predict(D[0])))

print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
