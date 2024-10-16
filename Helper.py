import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Callable, Tuple
import Assignment2.PlotHelper as ph


def generate_synthetic_data(
    func: Callable,
    data_range: Tuple[float, float] = (0.0, 20.0),
    n_samples: int = 100,
    noise_mean: float = 0.0,
    noise_variance: float = 1.0,
    noise_multiple: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param func: The function you want to generate synthetic data from :)
    :param data_range: a tuple of the range of data (i.e (0.0,20.0))
    :param n_samples: a number of samples (i.e. 100)
    :param noise_mean:
    :param noise_variance:
    :param noise_multiple: A value to multiply the noise with (i.e. 2.0)
    :return:  A tuple containing (x_values, y_values_with_noise, true_y_values)
    """
    # x_values = np.random.uniform(data_range[0], data_range[1], n_samples)
    x_values = np.linspace(data_range[0], data_range[1], n_samples)
    y_values = func(x_values)

    noise = np.random.normal(noise_mean, noise_variance, n_samples)

    y_values_noise = y_values + noise_multiple * noise

    return x_values, y_values_noise, y_values


def train_test_split(x, y_data, y_true, test_size=0.2, random_seed=None):
    """
    Splits the data into training and testing sets.

    :param x: Feature data
    :param y_data: Noisy Target data.
    :param y_true: True Target data.
    :param test_size: Proportion of the dataset to include in the test split (default: 0.2).
    :param random_seed: Seed for random number generator (default: None).
    :return: tuple: (x_train, x_test, y_train_noisy, y_test_noisy, y_train_true, y_test_true)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle the data
    indices = np.random.permutation(len(x))

    # Calculate the split index
    test_size = int(len(x) * test_size)
    train_size = len(x) - test_size

    # Split the data
    x_train, x_test = x[indices[:train_size]], x[indices[train_size:]]
    y_train, y_test = y_data[indices[:train_size]], y_data[indices[train_size:]]
    y_train_true, y_test_true = (
        y_true[indices[:train_size]],
        y_true[indices[train_size:]],
    )

    return x_train, x_test, y_train, y_test, y_train_true, y_test_true


def sinusoidal_function_for_synthetic_data(x):
    """
    function to generate sinusoidal function for synthetic data.
    :param x: variable x
    :return: sinusoidal function for synthetic data.
    """

    return np.sin(np.sqrt(x)) + np.cos(x) + np.sin(x)


def other_test_function_for_synthetic_data(x):
    return np.sin(x) + 0.5 * x * np.cos(x) + x


def linear_function_for_synthetic_data(x):
    """
    linear function for synthetic data.
    :param x: variable x
    :return: function for synthetic data.
    """
    return -4.0 * x + 10.0


def sigmoid(x, mu, sigma):
    return 1 / (1 + np.exp(-(x - mu) / sigma))


def gaussian(x, mu, sigma):
    """
    Gaussian function
    :param x: variable x
    :param mu: mean of gaussian function
    :param sigma: std of gaussian function
    :return: gaussian function
    """
    return np.exp(-(((x - mu) / sigma) ** 2))


def calculate_sse(y_true, y_pred):
    """
    Calculate the Sum of Squared Errors (SSE).

    :param y_true: Actual target values
    :param y_pred: Predicted target values
    :return:  SSE value
    """
    return np.sum((y_true - y_pred) ** 2)
