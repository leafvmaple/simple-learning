import numpy as np
import pandas as pd

class Linear:
    def __init__(self):
        self.learning_rate = 0.000011
        self.epochs        = 100000
        self.epsilon       = 0.00001

    def __cal_norm_params(self, X):
        self.mean_params = []
        self.std_params = []

        for i in range(X.shape[1]):
            self.mean_params.append(np.mean(X[:, i]))
            self.std_params.append(np.std(X[:, i]))

    def __error(self, X, y):
        return np.dot(X, self.w) - y

    def __euclidean_metric(self, v):
        return np.sqrt(np.sum(np.power(v, 2)))

    def __normalized(self, X):
        X_norm = []
        for i in range(X.shape[1]):
            if self.std_params[i] > 0:
                X_norm.append((X.T[i] - self.mean_params[i]) / self.std_params[i])
            else:
                X_norm.append(X.T[i])
        return np.array(X_norm).T

    def __train(self, X, y):
        for i in range(self.epochs):
            self.w = self.optimizer(X, y)
            if self.loss(X, y) < self.epsilon:
                break

            if self.verbose != 0 and i % 1000 == 0:
                print("epoch", i, "loss: ", self.loss(X, y))

    def fit(self, X, y, verbose=0):
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.w = np.zeros(self.n)
        self.verbose = verbose

        self.__cal_norm_params(X)

        self.X = self.__normalized(X)
        self.y = y

        self.__train(self.X, self.y)

    def predict(self, X):
        return np.dot(self.__normalized(X), self.w)

    def compile(self, loss='mse', optimizer='gd'):
        if loss == 'mse': # Mean Square Error
            self.loss = lambda X, y: np.sqrt(np.mean(np.power(self.__error(X, y), 2)))

        if optimizer == 'gd': # Gradient Descent
            self.optimizer = lambda X, y: self.w - np.dot(X.T, self.__error(X, y)) * self.learning_rate