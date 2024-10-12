import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from NonLinearRegression import *
from Assignment2.Helper import (
    generate_synthetic_data,
    sinusoidal_function_for_synthetic_data,
    gaussian,
    train_test_split,
    calculate_sse,
)
from Assignment2.PlotHelper import *


def model_fitting():

    data_range = (0.0, 20.0)
    n_samples = 100
    noise_mean = 0.0
    noise_variance = 1.0
    noise_multiple = 1.0

    # Generate 100 datapoints in range [0,20]
    x_values, y_values_noise, y_values = generate_synthetic_data(
        sinusoidal_function_for_synthetic_data,
        data_range,
        n_samples,
        noise_mean,
        noise_variance,
        noise_multiple,
    )

    # Plotting the data to predict
    plot_real_data_and_noisy_data(
        x_values,
        y_values_noise,
        y_values,
        output_folder="../Results",
        func=sinusoidal_function_for_synthetic_data,
    )

    for num_bases in range(0, 101, 10):

        lr = NonLinearRegression(False)

        # Compute basis matrix for the original data
        mu = np.linspace(0, 20, num_bases)
        phi = gaussian(x_values[:, None], mu[None, :], 1)

        # Fit the model on the original data
        lr.fit(phi, y_values_noise)

        plot_model_fit(
            lr,
            x_values,
            y_values_noise,
            mu,
            num_bases,
            gaussian,
            sinusoidal_function_for_synthetic_data,
            precision=10000,
            data_range=(0, 20),
            output_folder="../Results",
        )

    # Here we are simply plotting Gaussian bases


def gaussian_basis():
    precision = 1000  # number of data point per plots
    x = np.linspace(0, 20, precision)
    amount = 100  # number of
    mu = np.linspace(0, 20, amount)
    phi = gaussian(x[:, None], mu[None, :], 1)
    plot_gaussian_bases(x, phi, amount)


def sum_of_squared_errors():

    data_range = (0.0, 20.0)
    n_samples = 100
    noise_mean = 0.0
    noise_variance = 1.0
    noise_multiple = 1.0

    # Generate 100 datapoints in range [0,20]
    x_values, y_values_noise, y_values = generate_synthetic_data(
        sinusoidal_function_for_synthetic_data,
        data_range,
        n_samples,
        noise_mean,
        noise_variance,
        noise_multiple,
    )

    x_train, x_test, y_train, y_test, y_train_true, y_test_true = train_test_split(
        x_values, y_values_noise, y_values
    )

    plot_real_data_and_noisy_data(
        x_train,
        y_train,
        y_train_true,
        output_folder="../Results",
        func=sinusoidal_function_for_synthetic_data,
        filename="Train_Data_and_Noisy_Data_Distribution",
        graph_title="Synthetic Data Generation: True vs. Noisy Data",
    )

    sse_train_list = []
    sse_test_list = []
    range_of_value = range(0, 101, 5)
    for num_bases in range_of_value:
        lr = NonLinearRegression(False)

        # Compute basis matrix for the original data
        mu = np.linspace(0, 20, num_bases)
        phi_train = gaussian(x_train[:, None], mu[None, :], 1)
        phi_test = gaussian(x_test[:, None], mu[None, :], 1)

        # Fit the model on the training data
        lr.fit(phi_train, y_train)

        # Predict using the training and testing data
        y_train_pred = lr.predict(phi_train)
        y_test_pred = lr.predict(phi_test)

        # Calculate sum of squared errors for training and testing data
        sse_train = calculate_sse(y_train_true, y_train_pred)
        sse_test = calculate_sse(y_test_true, y_test_pred)

        print(
            f"Number of Bases: {num_bases}, SSE (Train): {sse_train}, SSE (Test): {sse_test}"
        )
        sse_train_list.append(sse_train)
        sse_test_list.append(sse_test)

    plot_sse(
        range_of_value,
        sse_test_list,
        output_folder="../Results",
        filename="SSE (Test)",
        title="Test - Sum of Squared Errors vs. Number of Bases",
    )
    plot_sse(
        range_of_value,
        sse_train_list,
        output_folder="../Results",
        filename="SSE (Train)",
        title="Train - Sum of Squared Errors vs. Number of Bases",
    )


if __name__ == "__main__":
    gaussian_basis()
    model_fitting()
    sum_of_squared_errors()