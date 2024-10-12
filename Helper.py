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

    :param func: a function to generate synthetic data
    :param data_range: a tuple of the range of data (i.e (0.0,20.0))
    :param n_samples: a number of samples (i.e. 100)
    :param noise_mean:
    :param noise_variance:
    :param noise_multiple: A value to multiply the noise with (i.e. 2.0)
    :return:  A tuple containing (x_values, y_values_with_noise, true_y_values)
    """
    x_values = np.random.uniform(data_range[0], data_range[1], n_samples)

    y_values = func(x_values)

    noise = np.random.normal(noise_mean, noise_variance, n_samples)

    y_values_noise = y_values + noise_multiple * noise

    return x_values, y_values_noise, y_values


def train_test_split(x, y_data, y_true, test_size=0.2, random_seed=None):
    """
    Splits the data into training and testing sets.

    Parameters:
    x: Feature data.
    y_data: Sample Target data.
    y_true: True Target data.
    test_size (float): Proportion of the dataset to include in the test split (default: 0.2).
    random_seed (int): Seed for random number generator (default: None).

    Returns:
    tuple: (x_train, x_test, y_train, y_test, y_train_true, y_test_true)
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
    return np.sin(np.sqrt(x)) + np.cos(x) + np.sin(x)


def linear_function_for_synthetic_data(x):
    return -4.0 * x + 10.0


def gaussian(x, mu, sigma):
    return np.exp(-(((x - mu) / sigma) ** 2))


def calculate_sse(y_true, y_pred):
    """
    Calculate the Sum of Squared Errors (SSE).

    Parameters:
    y_true: Actual target values
    y_pred: Predicted target values

    Returns:
    float: SSE value
    """
    return np.sum((y_true - y_pred) ** 2)
