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


# Linear Regression Model
class LR:
    def __init__(self, x, y, i=100000, lr=1e-3):
        self.num_item, self.num_feature = np.shape(x)
        self.X = np.ones((self.num_item, self.num_feature + 1))
        self.X[:, :-1] = x
        self.Y = y
        self.theta = np.ones(self.num_feature + 1)
        self.i = i
        self.lr = lr

    def fit(self, isOutput=True):
        x_T = self.X.transpose()
        for i in range(self.i):
            pred = np.dot(self.X, self.theta)
            tmp = pred - self.Y
            gradient = np.dot(x_T, tmp) / self.num_item
            self.theta = self.theta - self.lr * gradient
            cost = 1.0 / (2 * self.num_item) * np.sum(np.square(np.dot(self.X, self.theta) - self.Y))
            if isOutput:
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        res = []
        for x in x_list:
            x.append(1)
            res.append(np.dot(x, self.theta))
        return res

# Linear Regression with Normalization
class LRwN:
    def __init__(self, x, y, i=100000, lr=1e-3, alpha=1e-5):
        self.num_item, self.num_feature = np.shape(x)
        self.X = np.ones((self.num_item, self.num_feature + 1))
        self.X[:, :-1] = x
        self.Y = y
        self.theta = np.ones(self.num_feature + 1)
        self.i = i
        self.lr = lr
        self.alpha = alpha

    def fit(self, isOutput=True):
        x_T = self.X.transpose()
        for i in range(self.i):
            pred = np.dot(self.X, self.theta)
            tmp = pred - self.Y
            gradient = np.dot(x_T, tmp) / self.num_item
            self.theta = (1 - self.lr * self.alpha) * self.theta - self.lr * gradient
            cost = 1.0 / (2 * self.num_item) * np.sum(np.square(np.dot(self.X, self.theta) - self.Y))
            if isOutput:
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        res = []
        for x in x_list:
            x.append(1)
            res.append(np.dot(x, self.theta))
        return res

# Stochastic Gradient Descent
class SGD:
    def __init__(self, x, y, i=100000, lr=1e-3, alpha=1e-5):
        self.num_item, self.num_feature = np.shape(x)
        self.X = np.ones((self.num_item, self.num_feature + 1))
        self.X[:, :-1] = x
        self.Y = y
        self.theta = np.ones(self.num_feature + 1)
        self.i = i
        self.lr = lr
        self.alpha = alpha

    def fit(self, isOutput=True):
        l = list(range(self.num_item))
        random.shuffle(l)
        for i in range(self.i):
            p = l[i % len(l)]
            pred = np.dot(self.X[p], self.theta)
            tmp = pred - self.Y[p]
            gradient = np.dot(self.X[p], tmp)
            self.theta = (1 - self.lr * self.alpha) * self.theta - self.lr * gradient
            cost = 1.0 / 2.0 * np.square(np.dot(self.X[p], self.theta) - self.Y[p])
            if isOutput:
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        res = []
        for x in x_list:
            x.append(1)
            res.append(np.dot(x, self.theta))
        return res

# Mini-Batch Stochastic Gradient Descent
class MBSGD:
    def __init__(self, x, y, i=100000, lr=1e-3, alpha=1e-5, batch=4):
        self.num_item, self.num_feature = np.shape(x)
        self.X = np.ones((self.num_item, self.num_feature + 1))
        self.X[:, :-1] = x
        self.Y = y
        self.theta = np.ones(self.num_feature + 1)
        self.i = i
        self.lr = lr
        self.alpha = alpha
        self.batch = batch

    def fit(self, isOutput=True):
        l = list(range(self.num_item))
        random.shuffle(l)
        batch_size = self.num_item // self.batch if self.num_item > self.batch else 1
        x = []
        y = []
        for i in range(self.batch):
            a = []
            b = []
            for e in l[i * batch_size : (i + 1) * batch_size if i < self.batch - 1 else self.num_item]:
                a.append(self.X[e])
                b.append(self.Y[e])
            x.append(a)
            y.append(b)

        for i in range(self.i):
            xx = np.array(x[i % self.batch])
            yy = y[i % self.batch]
            pred = np.dot(xx, self.theta)
            tmp = pred - yy
            gradient = np.dot(xx.transpose(), tmp) / len(xx)
            self.theta = (1 - self.lr * self.alpha) * self.theta - self.lr * gradient
            cost = 1.0 / (2.0 * len(xx)) * np.sum(np.square(np.dot(xx, self.theta) - yy))
            if isOutput:
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        res = []
        for x in x_list:
            x.append(1)
            res.append(np.dot(x, self.theta))
        return res

class LR:
    def __init__(self, x, y, epoch=10000, lr=1e-3, alpha=0, batch=1):
        self.num_item, self.num_feature = np.shape(x)
        self.X = np.ones((self.num_item, self.num_feature + 1))
        self.X[:, :-1] = x
        self.Y = y
        self.theta = np.ones(self.num_feature + 1)
        self.epoch = epoch
        self.lr = lr
        self.alpha = alpha
        self.batch = batch

    def shuffle(self):
        p = np.random.permutation(len(self.Y))
        return self.X[p], np.array(self.Y, dtype=float)[p]

    def get_batch(self, x, y, batch_size=50):
        ret = []
        cnt = self.num_item // batch_size if self.num_item % batch_size == 0 else self.num_item // batch_size + 1
        for i in range(cnt):
            st = i * batch_size
            ed = (i + 1) * batch_size if (i + 1) * batch_size < self.num_item else self.num_item
            ret.append([x[st:ed], y[st:ed]])
        return ret

    def fit(self):
        batch_size = self.num_item // self.batch if self.num_item % self.batch == 0 else self.num_item // self.batch + 1
        for i in range(self.epoch):
            x, y = self.shuffle()
            for batch in self.get_batch(x, y, batch_size):
                batch_x, batch_y = batch[0], batch[1].tolist()
                pred = np.dot(batch_x, self.theta)
                tmp = pred - batch_y
                grad = np.dot(batch_x.transpose(), tmp) / len(batch_y) + self.alpha * self.theta
                self.theta = self.theta - self.lr * grad
                cost = 1.0 / (2.0 * len(batch_y)) * np.sum(np.square(np.dot(batch_x, self.theta) - batch_y))
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        ret = []
        for x in x_list:
            x.append(1)
            ret.append(np.dot(x, self.theta))
        return ret

class LRM:
    def __init__(self, x, y, epoch=10000, lr=1e-3, alpha=0, gamma=0.9, batch=1):
        self.num_item, self.num_feature = np.shape(x)
        self.X = np.ones((self.num_item, self.num_feature + 1))
        self.X[:, :-1] = x
        self.Y = y
        self.theta = np.ones(self.num_feature + 1)
        self.epoch = epoch
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.batch = batch


    def shuffle(self):
        p = np.random.permutation(len(self.Y))
        return self.X[p], np.array(self.Y, dtype=float)[p]

    def get_batch(self, x, y, batch_size=50):
        ret = []
        cnt = self.num_item // batch_size if self.num_item % batch_size == 0 else self.num_item // batch_size + 1
        for i in range(cnt):
            st = i * batch_size
            ed = (i + 1) * batch_size if (i + 1) * batch_size < self.num_item else self.num_item
            ret.append([x[st:ed], y[st:ed]])
        return ret

    def fit(self):
        batch_size = self.num_item // self.batch if self.num_item % self.batch == 0 else self.num_item // self.batch + 1
        v = np.zeros(self.num_feature + 1)
        for i in range(self.epoch):
            x, y = self.shuffle()
            for batch in self.get_batch(x, y, batch_size):
                batch_x, batch_y = batch[0], batch[1].tolist()
                pred = np.dot(batch_x, self.theta)
                tmp = pred - batch_y
                grad = np.dot(batch_x.transpose(), tmp) / len(batch_y) + self.alpha * self.theta
                v = self.gamma * v + (1 - self.gamma) * grad * self.lr
                self.theta = self.theta - v
                cost = 1.0 / (2.0 * len(batch_y)) * np.sum(np.square(np.dot(batch_x, self.theta) - batch_y))
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        ret = []
        for x in x_list:
            x.append(1)
            ret.append(np.dot(x, self.theta))
        return ret

class Adagrad:
    def __init__(self, x, y, epoch=10000, lr=1e-3, alpha=0, batch=1):
        self.num_item, self.num_feature = np.shape(x)
        self.X = np.ones((self.num_item, self.num_feature + 1))
        self.X[:, :-1] = x
        self.Y = y
        self.theta = np.ones(self.num_feature + 1)
        self.epoch = epoch
        self.lr = lr
        self.alpha = alpha
        self.batch = batch


    def shuffle(self):
        p = np.random.permutation(len(self.Y))
        return self.X[p], np.array(self.Y, dtype=float)[p]

    def get_batch(self, x, y, batch_size=50):
        ret = []
        cnt = self.num_item // batch_size if self.num_item % batch_size == 0 else self.num_item // batch_size + 1
        for i in range(cnt):
            st = i * batch_size
            ed = (i + 1) * batch_size if (i + 1) * batch_size < self.num_item else self.num_item
            ret.append([x[st:ed], y[st:ed]])
        return ret

    def fit(self):
        batch_size = self.num_item // self.batch if self.num_item % self.batch == 0 else self.num_item // self.batch + 1
        v = np.square(self.theta)
        for i in range(self.epoch):
            x, y = self.shuffle()
            for batch in self.get_batch(x, y, batch_size):
                batch_x, batch_y = batch[0], batch[1].tolist()
                pred = np.dot(batch_x, self.theta)
                tmp = pred - batch_y
                grad = np.dot(batch_x.transpose(), tmp) / len(batch_y) + self.alpha * self.theta
                self.theta = self.theta - self.lr * grad / np.sqrt(v)
                v = v + np.square(self.theta)
                cost = 1.0 / (2.0 * len(batch_y)) * np.sum(np.square(np.dot(batch_x, self.theta) - batch_y))
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        ret = []
        for x in x_list:
            x.append(1)
            ret.append(np.dot(x, self.theta))
        return ret

class RMSprop:
    def __init__(self, x, y, epoch=10000, lr=1e-3, alpha=0, gamma=0.9, batch=1):
        self.num_item, self.num_feature = np.shape(x)
        self.X = np.ones((self.num_item, self.num_feature + 1))
        self.X[:, :-1] = x
        self.Y = y
        self.theta = np.ones(self.num_feature + 1)
        self.epoch = epoch
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.batch = batch


    def shuffle(self):
        p = np.random.permutation(len(self.Y))
        return self.X[p], np.array(self.Y, dtype=float)[p]

    def get_batch(self, x, y, batch_size=50):
        ret = []
        cnt = self.num_item // batch_size if self.num_item % batch_size == 0 else self.num_item // batch_size + 1
        for i in range(cnt):
            st = i * batch_size
            ed = (i + 1) * batch_size if (i + 1) * batch_size < self.num_item else self.num_item
            ret.append([x[st:ed], y[st:ed]])
        return ret

    def fit(self):
        batch_size = self.num_item // self.batch if self.num_item % self.batch == 0 else self.num_item // self.batch + 1
        v = np.square(self.theta)
        for i in range(self.epoch):
            x, y = self.shuffle()
            for batch in self.get_batch(x, y, batch_size):
                batch_x, batch_y = batch[0], batch[1].tolist()
                pred = np.dot(batch_x, self.theta)
                tmp = pred - batch_y
                grad = np.dot(batch_x.transpose(), tmp) / len(batch_y) + self.alpha * self.theta
                self.theta = self.theta - self.lr * grad / np.sqrt(v)
                v = self.gamma * v + (1 - self.gamma) * np.square(self.theta)
                cost = 1.0 / (2.0 * len(batch_y)) * np.sum(np.square(np.dot(batch_x, self.theta) - batch_y))
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        ret = []
        for x in x_list:
            x.append(1)
            ret.append(np.dot(x, self.theta))
        return ret

class Adam:
    def __init__(self, x, y, epoch=10000, lr=1e-3, alpha=0, gamma=0.9, beta=0.9, batch=1):
        self.num_item, self.num_feature = np.shape(x)
        self.X = np.ones((self.num_item, self.num_feature + 1))
        self.X[:, :-1] = x
        self.Y = y
        self.theta = np.ones(self.num_feature + 1)
        self.epoch = epoch
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.batch = batch

    def shuffle(self):
        p = np.random.permutation(len(self.Y))
        return self.X[p], np.array(self.Y, dtype=float)[p]

    def get_batch(self, x, y, batch_size=50):
        ret = []
        cnt = self.num_item // batch_size if self.num_item % batch_size == 0 else self.num_item // batch_size + 1
        for i in range(cnt):
            st = i * batch_size
            ed = (i + 1) * batch_size if (i + 1) * batch_size < self.num_item else self.num_item
            ret.append([x[st:ed], y[st:ed]])
        return ret

    def fit(self):
        batch_size = self.num_item // self.batch if self.num_item % self.batch == 0 else self.num_item // self.batch + 1
        m = np.zeros(self.num_feature + 1)
        v = np.zeros(self.num_feature + 1)
        for i in range(self.epoch):
            x, y = self.shuffle()
            for batch in self.get_batch(x, y, batch_size):
                batch_x, batch_y = batch[0], batch[1].tolist()
                pred = np.dot(batch_x, self.theta)
                tmp = pred - batch_y
                grad = np.dot(batch_x.transpose(), tmp) / len(batch_y) + self.alpha * self.theta
                m = self.lr * (self.gamma * m + (1 - self.gamma) * grad)
                v = self.beta * v + (1 - self.beta) * np.square(grad)
                m_bias = m / (1 - self.gamma ** (i + 1))
                v_bias = v / (1 - self.beta ** (i + 1))
                self.theta = self.theta - 1.0 / np.sqrt(v_bias) * m_bias
                cost = 1.0 / (2.0 * len(batch_y)) * np.sum(np.square(np.dot(batch_x, self.theta) - batch_y))
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        ret = []
        for x in x_list:
            x.append(1)
            ret.append(np.dot(x, self.theta))
        return ret