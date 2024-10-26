import numpy as np


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


class LinearRegression:
    def __init__(self, add_bias=True, l2_reg=0, l1_reg=0):
        """
        :param add_bias: Boolean to indicate whether to add a bias term
        :param l2_reg: L2 regularization strength (default 0, meaning no L2 regularization)
        :param l1_reg: L1 regularization strength (default 0, meaning no L1 regularization)
        """
        self.add_bias = add_bias
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg

    def fit(self, x, y, optimizer):
        if x.ndim == 1:
            x = x[:, None]  # Reshape if input is a 1D array
        if self.add_bias:
            n = x.shape[0]
            x = np.column_stack([x, np.ones(n)])  # Add bias term (column of ones)

        n, d = x.shape

        # Gradient function with L1 and L2 regularization, excluding bias term
        def gradient(x, y, w):
            yh = x @ w  # Predictions
            grad = 0.5 * np.dot(yh - y, x) / n  # Gradient of the loss (MSE)

            # Apply L2 regularization to non-bias coefficients
            if self.l2_reg > 0:
                grad[1:] += self.l2_reg * w[1:]

            # Apply L1 regularization to non-bias coefficients
            if self.l1_reg > 0:
                grad[1:] += self.l1_reg * np.sign(w[1:])

            return grad

        # Initialize weights
        w0 = np.zeros(d)

        # Use the optimizer to minimize the loss and find the weights
        self.w = optimizer.run(gradient, x, y, w0)

        return self

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]  # Reshape if input is a 1D array
        if self.add_bias:
            n = x.shape[0]
            x = np.column_stack([x, np.ones(n)])  # Add bias term (column of ones)

        yh = x @ self.w  # Predict the target values
        return yh
