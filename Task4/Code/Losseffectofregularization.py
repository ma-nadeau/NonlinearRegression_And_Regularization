import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from Assignment2.PlotHelper import *

N = 50
x = np.linspace(0, 10, N)

epsilon = np.random.normal(0, 1, N)

y = -4 * x + 10 + 2 * epsilon

# Plot the generated data
'''plt.scatter(x, y, label="Synthetic Data", color='blue')
plt.title("y = -4x + 10 + 2ε")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()'''


class GradientDescent:

    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8, record_history=False):
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

        w0 = np.zeros(D)
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

        w0 = np.zeros(D)
        self.w = optimizer.run(gradient, x, y, w0)
        return self

    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([x, np.ones(N)])
        yh = x @ self.w
        return yh


l2_penalty = lambda w: np.dot(w, w) / 2
l1_penalty = lambda w: np.sum(np.abs(w))
plot_cost_func_contour(x, y)

cost2 = lambda w, reg: .5 * np.mean((w[0] + w[1] * x - y) ** 2) + reg * l2_penalty(w)
reg_list = [0, 1, 2, 3, 4, 5]

for i, reg_coef in enumerate(reg_list):
    fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(15, 5))
    optimizer = GradientDescent(learning_rate=.01, max_iters=1000, record_history=True)
    model = L2_Regression(optimizer, l2_reg=reg_coef)
    model.fit(x, y, optimizer)
    current_cost = lambda w: cost2(w, reg_coef)
    plot_gradient_descent(current_cost, optimizer, axes, reg_coef)

cost1 = lambda w, reg: .5 * np.mean((w[0] + w[1] * x - y) ** 2) + reg * l1_penalty(w)
for i, reg_coef in enumerate(reg_list):
    fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(15, 5))
    optimizer = GradientDescent(learning_rate=.01, max_iters=1000, record_history=True)
    model = L1_Regression(optimizer, l1_reg=reg_coef)
    model.fit(x, y, optimizer)
    current_cost = lambda w: cost1(w, reg_coef)
    plot_gradient_descent(current_cost, optimizer, axes, reg_coef, False)
