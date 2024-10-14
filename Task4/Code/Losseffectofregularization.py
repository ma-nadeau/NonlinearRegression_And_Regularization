import numpy as np
import matplotlib.pyplot as plt
import itertools

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

def plot_contour(f, x1bound, x2bound, resolution, ax):
    x1range = np.linspace(x1bound[0], x1bound[1], resolution)
    x2range = np.linspace(x2bound[0], x2bound[1], resolution)
    xg, yg = np.meshgrid(x1range, x2range)
    zg = np.zeros_like(xg)
    for i,j in itertools.product(range(resolution), range(resolution)):
        zg[i,j] = f([xg[i,j], yg[i,j]])
    ax.contour(xg, yg, zg, 100)
    return ax

cost = lambda w: .5*np.mean((w[0] + w[1]*x - y)**2)
l2_penalty = lambda w: np.dot(w,w)/2
l1_penalty = lambda w: np.sum(np.abs(w))
strength = [5]
fig, axes = plt.subplots(ncols=2+len(strength), nrows=1, constrained_layout=True, figsize=(15, 5))
plot_contour(cost, [-20,20], [-20,20], 50, axes[0])
axes[0].set_title(r'cost function $J(w)$')
plot_contour(l2_penalty, [-20,20], [-20,20], 50, axes[1])
axes[1].set_title(r'L2 reg. $||w||_2^2$')
for i in range( len(strength)):
    cost_plus_l2 = lambda w: cost(w) + strength[i]*l2_penalty(w)
    plot_contour(cost_plus_l2, [-20,20], [-20,20], 50, axes[i+2])
    axes[i+2].set_title(r'L2 reg. cost $J(w) + '+str(strength[i])+' ||w||_2^2$')


fig, axes = plt.subplots(ncols=2+len(strength), nrows=1, constrained_layout=True, figsize=(15, 5))
plot_contour(cost, [-20,20], [-20,20], 50, axes[0])
axes[0].set_title(r'cost function $J(w)$')
plot_contour(l1_penalty, [-20,20], [-20,20], 50, axes[1])
axes[1].set_title(r'L1 reg. $||w||$')
for i in range( len(strength)):
    cost_plus_l1 = lambda w: cost(w) + strength[i]*l1_penalty(w)
    plot_contour(cost_plus_l1, [-20,20], [-20,20], 50, axes[i+2])
    axes[i+2].set_title(r'L1 reg. cost $J(w) + '+str(strength[i])+' ||w||$')


cost2 = lambda w, reg: .5*np.mean((w[0] + w[1]*x - y)**2) + reg*l2_penalty(w)
reg_list = [0, 5, 30]
fig, axes = plt.subplots(ncols=len(reg_list), nrows=1, constrained_layout=True, figsize=(15, 5))
for i, reg_coef in enumerate(reg_list):
    optimizer = GradientDescent(learning_rate=.01, max_iters=50, record_history=True)
    model = L2_Regression(optimizer, l2_reg=reg_coef)
    model.fit(x,y, optimizer)
    current_cost = lambda w: cost2(w, reg_coef)
    plot_contour(current_cost, [-20,20], [-5,5], 50, axes[i])
    w_hist = np.vstack(optimizer.w_history)# T x 2
    axes[i].plot(w_hist[:,1], w_hist[:,0], '.r', alpha=.8)
    axes[i].plot(w_hist[:,1], w_hist[:,0], '-r', alpha=.3)
    axes[i].set_xlabel(r'$w_0$')
    axes[i].set_ylabel(r'$w_1$')
    axes[i].set_title(f' lambda = {reg_coef}')
    axes[i].set_xlim([-20,20])
    axes[i].set_ylim([-5,5])

cost1 = lambda w, reg: .5*np.mean((w[0] + w[1]*x - y)**2) + reg*l1_penalty(w)
reg_list = [0, 5,30]
fig, axes = plt.subplots(ncols=len(reg_list), nrows=1, constrained_layout=True, figsize=(15, 5))
for i, reg_coef in enumerate(reg_list):
    optimizer = GradientDescent(learning_rate=.01, max_iters=50, record_history=True)
    model = L1_Regression(optimizer, l1_reg=reg_coef)
    model.fit(x,y, optimizer)
    current_cost = lambda w: cost1(w, reg_coef)
    plot_contour(current_cost, [-20,20], [-5,5], 50, axes[i])
    w_hist = np.vstack(optimizer.w_history)# T x 2
    axes[i].plot(w_hist[:,1], w_hist[:,0], '.r', alpha=.8)
    axes[i].plot(w_hist[:,1], w_hist[:,0], '-r', alpha=.3)
    axes[i].set_xlabel(r'$w_0$')
    axes[i].set_ylabel(r'$w_1$')
    axes[i].set_title(f' lambda = {reg_coef}')
    axes[i].set_xlim([-20,20])
    axes[i].set_ylim([-5,5])
    plt.savefig(f'plot_lambda_{reg_coef}.png')  # Save the figure to a file
    plt.close()