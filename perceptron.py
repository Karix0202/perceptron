import numpy as np

class Perceptron(object):
    def __init__(self, eta=.01, n_iter=50, init_with_zeros=True, random_seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.init_with_zeros = init_with_zeros
        self.random_seed = random_seed

        self.errors = None
        self.theta = None

    def init_theta(self, X):
        shape = (X.shape[1] + 1, 1)

        if self.init_with_zeros:
            self.theta = np.zeros(shape)
        else:
            rnd_gen = np.random.RandomState(self.random_seed)
            self.theta = rnd_gen.normal(size=shape, loc=0, scale=.01)

    def add_intercept(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def fit(self, X, y):
        self.init_theta(X)

        y = y.reshape(-1, 1)
        X = self.add_intercept(X)

        for i in range(self.n_iter):
            self.theta += self.eta * np.dot(X.T, (y - self.predict(X)))

    def output(self, X):
        return np.dot(X, self.theta)

    def predict(self, X):
        if X.shape[1] != self.theta.shape[0]:
            X = self.add_intercept(X)
        return np.where(self.output(X) >= 0, 1, -1)
