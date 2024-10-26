import matplotlib.pyplot as plt
import itertools
import os

import numpy as np

from LinearRegressionNonLinearBasisFunctions.Code.PlotHelper import *
from Gradientdescent_regressions import *

N = 50
x = np.random.uniform(0, 10, N)

epsilon = np.random.normal(0, 1, N)

y = 10 - 4 * x + 2 * epsilon

plot_T4_syth_data(x, y)

l2_penalty = lambda w: np.dot(w, w) / 2
l1_penalty = lambda w: np.sum(np.abs(w))
plot_cost_func_contour(x, y)

cost2 = lambda w, reg: 0.5 * np.mean((w[1] + w[0] * x - y) ** 2) + reg * l2_penalty(
    w[0]
)
reg_list = list(range(50))

w_vs_lam_l2 = []
for i, reg_coef in enumerate(reg_list):
    fig, axes = plt.subplots(
        ncols=1, nrows=1, constrained_layout=True, figsize=(10, 10)
    )
    optimizer = GradientDescent(
        learning_rate=0.001, max_iters=10000, record_history=True
    )
    model = L2_Regression(optimizer, l2_reg=reg_coef)
    model.fit(x, y, optimizer)
    current_cost = lambda w: cost2(w, reg_coef)
    w_vs_lam_l2.append(optimizer.w_history[-1])
    plot_gradient_descent(current_cost, optimizer, axes, reg_coef)

w_vs_lam_l1 = []

cost1 = lambda w, reg: 0.5 * np.mean((w[1] + w[0] * x - y) ** 2) + reg * l1_penalty(
    w[0]
)
for i, reg_coef in enumerate(reg_list):
    fig, axes = plt.subplots(
        ncols=1, nrows=1, constrained_layout=True, figsize=(10, 10)
    )
    optimizer = GradientDescent(
        learning_rate=0.001, max_iters=10000, record_history=True
    )
    model = L1_Regression(optimizer, l1_reg=reg_coef)
    model.fit(x, y, optimizer)
    current_cost = lambda w: cost1(w, reg_coef)
    w_vs_lam_l1.append(optimizer.w_history[-1])
    plot_gradient_descent(current_cost, optimizer, axes, reg_coef, False)


plot_weight_vs_lambda(reg_list, w_vs_lam_l1, w_vs_lam_l2)
