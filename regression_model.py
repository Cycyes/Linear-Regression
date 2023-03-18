import numpy as np

# Linear Regression
class LR:
    def __init__(self, x, y, epoch=10000, lr=1e-3, alpha=0, gamma=0.9, beta=0.9, batch=1, optimizer="None"):
        '''
        try:
            self.num_item, self.num_feature = np.shape(x)
        except:
            self.num_item = np.shape(x)[0]
            self.num_feature = 1
        '''
        try:
            self.num_item, self.num_feature = np.shape(x)
        except:
            self.num_item = np.shape(x)[0]
            self.num_feature = 1
            x = np.array(x).reshape(self.num_item, 1)
            print(x)
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
        self.optimizer = optimizer

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

    def judge(self, v_bias, eps):
        for v in v_bias:
            if v < eps:
                return False
        return True

    def fit(self):
        batch_size = self.num_item // self.batch if self.num_item % self.batch == 0 else self.num_item // self.batch + 1
        m = np.zeros(self.num_feature + 1)
        M = np.zeros(self.num_feature + 1)
        v = np.zeros(self.num_feature + 1)
        for i in range(self.epoch):
            x, y = self.shuffle()
            for batch in self.get_batch(x, y, batch_size):
                batch_x, batch_y = batch[0], batch[1].tolist()
                pred = np.dot(batch_x, self.theta)
                tmp = pred - batch_y
                grad = np.dot(batch_x.transpose(), tmp) / len(batch_y) + self.alpha * self.theta

                m_bias = m / (1 - self.gamma ** (i + 1))
                v_bias = v / (1 - self.beta ** (i + 1))

                if self.optimizer == "None":
                    self.theta = self.theta - self.lr * grad
                elif self.optimizer == "Momentum":
                    M = self.gamma * M + (1 - self.gamma) * grad * self.lr
                    self.theta = self.theta - M
                elif self.optimizer == "RMSprop":
                    if self.judge(v_bias, 1e-5):
                        self.theta = self.theta - self.lr * grad / (np.sqrt(v_bias))
                    else:
                        self.theta = self.theta - self.lr * grad
                elif self.optimizer == "Adam":
                    if self.judge(v_bias, 1e-5):
                        self.theta = self.theta - 1.0 / np.sqrt(v_bias) * m_bias
                    else:
                        self.theta = self.theta - self.lr * grad

                m = self.lr * (self.gamma * m + (1 - self.gamma) * grad)
                v = self.beta * v + (1 - self.beta) * np.square(grad)

                cost = 1.0 / (2.0 * len(batch_y)) * np.sum(np.square(np.dot(batch_x, self.theta) - batch_y))
                print("the {}th cost is: {}".format(i, round(cost, 2)))
        return self

    def predict(self, x_list):
        ret = []
        for x in x_list:
            x.append(1)
            ret.append(np.dot(x, self.theta))
        return ret

    def mse(self, x, y):
        y_pred = self.predict(x)
        return np.mean((y - y_pred) ** 2)

    def rmse(self, x, y):
        y_pred = self.predict(x)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def mae(self, x, y):
        y_pred = self.predict(x)
        return np.mean(np.abs(y - y_pred))

    def r_squared(self, x, y):
        y_pred = self.predict(x)
        mse = np.sum((y - y_pred) ** 2)
        var = np.sum((y - np.mean(y)) ** 2)
        return 1 - (mse / var)

