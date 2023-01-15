import numpy as np
import matplotlib.pyplot as plt


class TaskPredict:

    def __init__(self, x):
        self.A = np.identity(5)
        self.b = np.zeros(5).reshape((-1, 1))
        self.x = np.array(x).reshape((-1, 1))
        self.gama = 1
        self.min = float('inf')
        self.max = float('-inf')

    def update_params(self, reward):
        if self.min > reward:
            self.min = reward
        if self.max < reward:
            self.max = reward
        self.A = self.A + np.dot(self.x, self.x.reshape(1, -1))
        # gap = (self.max - self.min) * 0.5
        # self.b = self.b + (reward - gap) * self.x
        self.b = self.b + reward * self.x

    def predict_task(self):
        A_inverse = np.linalg.inv(self.A)
        theta = np.dot(A_inverse, self.b)
        task = np.dot(theta.T, self.x) + self.gama * np.dot(np.dot(self.x.reshape(1, -1), A_inverse), self.x) ** 0.5
        return task

    def test(self, m):
        m = np.array(m).reshape((-1, 1))
        A_inverse = np.linalg.inv(self.A)
        theta = np.dot(A_inverse, self.b)
        task = np.dot(theta.T, m) + self.gama * np.dot(np.dot(m.reshape(1, -1), A_inverse), m) ** 0.5
        return task

    def get_first(self):
        A_inverse = np.linalg.inv(self.A)
        theta = np.dot(A_inverse, self.b)
        first = np.dot(theta.T, self.x)
        return first

    def init_param(self, reward):
        self.min = reward
        self.max = reward
        self.A = self.A + np.dot(self.x, self.x.reshape(1, -1))
        self.b = self.b + (reward - (self.max - self.min)) * self.x


if __name__ == '__main__':
    a = TaskPredict([0.5, 0.5, 0.5, 0.5, 1.])
    real = np.zeros(1000)
    predict = np.zeros(1000)
    first = np.zeros(1000)
    task_real = np.random.randint(600, 700, 1)[0]
    a.init_param(task_real)
    num = 0
    num_failure = 0
    min = 20000
    max = 0
    for i in range(1000):
        predict[i] = a.predict_task()
        first[i] = a.get_first()
        task_real = np.random.randint(600, 700, 1)[0]
        a.update_params(task_real)
        real[i] = task_real
        if predict[i] > real[i]:
            # 除数值为：下限除以区间大小乘以10 最好
            # a.update_params(predict[i]/((min / (max - min + 1)) * 100))
            num_failure = num_failure + 1
            a.update_params(predict[i]/(num_failure * 2))
        else:
            num = num + 1
            a.update_params(task_real)
            if min > task_real:
                min = task_real
            if max < task_real:
                max = task_real
    print(a.test([0.5, 0.5, 0.5, 0.5, 1.]))
    print(num)
    print(predict[0], predict[1])
    plt.figure()
    x = range(1000)
    plt.plot(x, real)
    plt.plot(x, predict)
    plt.plot(x, first)
    plt.legend(['real', 'predict', 'first'])
    plt.show()
