import numpy as np


class NonLinearRegression:
    def __init__(self, add_bias=False):
        self.weights = None
        self.add_bias = add_bias

    def fit(self, X, Y):
        """
        From the matrix X and vector Y, fit computes the weights for the linear regression
        :param X: matrix containing the features
        :param Y: vector containing the target
        :return: self
        """
        # Add a column of 1 to X -> [1, x1, x2, ..., x_D]^T
        if self.add_bias:
            X = np.c_[np.ones(X.shape[0]), X]  # c_ -> concatenates along second axis
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ Y
        return self

    def predict(self, X):
        """
        Once a model has been trained, we predict y_hat using the weights and the matrix X
        :param X: matrix containing the features
        :return: y_hat (vector): A vector containing the predicted targets
        """
        # Add a column of 1 to X -> [1, x1, x2, ..., x_D]^T
        if self.add_bias:
            X = np.c_[np.ones(X.shape[0]), X]  # c_ -> concatenates along second axis
        y_hat = X @ self.weights  # y_hat = X * w
        return y_hat  # Returns the prediction made by our model

    def get_weights(self):
        """
        Returns the weights of the linear regression
        :return: weights of linear regression
        """
        return self.weights
