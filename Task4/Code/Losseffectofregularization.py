
import matplotlib.pyplot as plt
import itertools
import os
from Assignment2.PlotHelper import *
from Gradientdescent_regressions import *
N = 50
x = np.linspace(0, 10, N)

epsilon = np.random.normal(0, 1, N)

y = -4 * x + 10 + 2 * epsilon

plot_T4_syth_data(x,y)

l2_penalty = lambda w: np.dot(w, w) / 2
l1_penalty = lambda w: np.sum(np.abs(w))
plot_cost_func_contour(x, y)

cost2 = lambda w, reg: .5 * np.mean((w[0] + w[1] * x - y) ** 2) + reg * l2_penalty(w)
reg_list = [0, 1, 2, 3, 4, 5]

for i, reg_coef in enumerate(reg_list):
    fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(10, 10))
    optimizer = GradientDescent(learning_rate=.01, max_iters=1000, record_history=True)
    model = L2_Regression(optimizer, l2_reg=reg_coef)
    model.fit(x, y, optimizer)
    current_cost = lambda w: cost2(w, reg_coef)
    plot_gradient_descent(current_cost, optimizer, axes, reg_coef)

cost1 = lambda w, reg: .5 * np.mean((w[0] + w[1] * x - y) ** 2) + reg * l1_penalty(w)
for i, reg_coef in enumerate(reg_list):
    fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(10, 10))
    optimizer = GradientDescent(learning_rate=.01, max_iters=1000, record_history=True)
    model = L1_Regression(optimizer, l1_reg=reg_coef)
    model.fit(x, y, optimizer)
    current_cost = lambda w: cost1(w, reg_coef)
    plot_gradient_descent(current_cost, optimizer, axes, reg_coef, False)
