import numpy as np
N = 50
class GradientDescent:

    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-7, record_history=False):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.record_history = record_history
        self.epsilon = epsilon
        if record_history:
            self.w_history = []

    def run(self, gradient_fn, x, y, w):
        grad = np.inf
        t = 1
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            grad = gradient_fn(x, y, w)
            w = w - self.learning_rate * grad
            if self.record_history:
                self.w_history.append(w)
            t += 1
        return w


class L2_Regression:
    def __init__(self, add_bias=True, l2_reg=0):
        self.add_bias = add_bias
        self.l2_reg = l2_reg
        pass

    def fit(self, x, y, optimizer):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])
        N, D = x.shape

        def gradient(x, y, w):
            yh = x @ w
            N, D = x.shape
            grad = .5 * np.dot(yh - y, x) / N
            grad += self.l2_reg * w
            return grad

        w0 = np.ones(D)
        self.w = optimizer.run(gradient, x, y, w0)
        return self

    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([x, np.ones(N)])
        yh = x @ self.w
        return yh


class L1_Regression:
    def __init__(self, add_bias=True, l1_reg=0):
        self.add_bias = add_bias
        self.l1_reg = l1_reg
        pass

    def fit(self, x, y, optimizer):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])
        N, D = x.shape

        def gradient(x, y, w):
            yh = x @ w
            N, D = x.shape
            grad = .5 * np.dot(yh - y, x) / N
            grad += self.l1_reg * np.sign(w)
            return grad

        w0 = np.ones(D)
        self.w = optimizer.run(gradient, x, y, w0)
        return self

    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([x, np.ones(N)])
        yh = x @ self.w
        return yh

