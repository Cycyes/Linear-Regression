import numpy as np

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

    def fit(self, isOutput):
        x_T = self.X.transpose()
        for i in range(self.i):
            pred = np.dot(self.X, self.theta)
            tmp = pred - self.Y
            gradient = np.dot(x_T, tmp) / self.num_item
            self.theta = self.theta - self.lr * gradient
            cost = 1.0 / (2 * self.num_item) * np.sum(np.square(np.dot(self.X, self.theta) - self.Y))
            if isOutput:
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self.theta

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

    def fit(self, isOutput):
        x_T = self.X.transpose()
        for i in range(self.i):
            pred = np.dot(self.X, self.theta)
            tmp = pred - self.Y
            gradient = np.dot(x_T, tmp) / self.num_item
            self.theta = (1 - self.lr * self.alpha) * self.theta - self.lr * gradient
            cost = 1.0 / (2 * self.num_item) * np.sum(np.square(np.dot(self.X, self.theta) - self.Y))
            if isOutput:
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self.theta

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

    def fit(self, isOutput):
        x_T = self.X.transpose()
        for i in range(self.i):
            pred = np.dot(self.X, self.theta)
            tmp = pred - self.Y
            gradient = np.dot(x_T, tmp) / self.num_item
            self.theta = (1 - self.lr * self.alpha) * self.theta - self.lr * gradient
            cost = 1.0 / (2 * self.num_item) * np.sum(np.square(np.dot(self.X, self.theta) - self.Y))
            if isOutput:
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self.theta

    def predict(self, x_list):
        res = []
        for x in x_list:
            x.append(1)
            res.append(np.dot(x, self.theta))
        return res



