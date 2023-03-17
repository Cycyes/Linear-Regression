import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1, 5.56], [2, 5.70], [3, 5.91], [4, 6.40], [5, 6.80], [6, 7.05], [7, 8.90], [8, 8.70], [9, 9.00], [10, 9.05]])
num_item, num_feature = np.shape(data)

def data_process(data):
    x = np.ones((num_item, num_feature)) # 初始化 x
    x[:, :-1] = data[:, :-1]
    y = data[:, -1] # 赋值 y
    theta = np.ones(num_feature) # 初始化参数
    return x, y, theta

def train(iter, x, y, theta, lr):
    x_T = x.transpose()
    for i in range(iter):
        pred = np.dot(x, theta)
        temp = pred - y
        gradient = np.dot(x_T, temp) / num_item # 梯度
        theta = theta - lr * gradient
        cost = 1.0 / (2 * num_item) * np.sum(np.square(np.dot(x, theta) - y))
        print("第{}次梯度下降损失为：{}".format(i, round(cost, 2)))
    return theta

def train_norm(iter, x, y, theta, lr, r):
    x_T = x.transpose()
    for i in range(iter):
        pred = np.dot(x, theta)
        temp = pred - y
        gradient = np.dot(x_T, temp) / num_item  # 梯度
        theta = theta - lr * gradient - lr * r * theta
        cost = 1.0 / (2 * num_item) * np.sum(np.square(np.dot(x, theta) - y))
        print("第{}次norm梯度下降损失为：{}".format(i, round(cost, 2)))
    return theta


def predict(x, theta):
    return np.dot(x, theta)

x, y, theta = data_process(data)

w = train(1000, x, y, theta, 0.01)
w_norm = train_norm(1000, x, y, theta, 0.01, 0.005)
print(predict(x, w))
print(predict(x, w_norm))

plt.title("regression model")
plt.scatter(data[:, 0], data[:, -1], color='b', label='train')
plt.plot(data[:,0], predict(x, w_norm), color='r', label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()