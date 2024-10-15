import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from typing import Callable, Tuple, Optional
from Assignment2.Helper import *


def plot_real_data_and_noisy_data(
    x: np.array,
    y_noisy: np.array,
    y_real: np.array,
    output_folder: str = "../Results",
    filename: str = "Data_and_Noisy_Data_Distribution",
    graph_title: str = "Synthetic Data Generation: True vs. Noisy Data",
    func: Optional[Callable] = None,
):
    """
    Plots and saves a graph comparing the real data with the noisy data.

    :param x: A numpy array representing the x-values.
    :param y_noisy: A numpy array representing the noisy y-values.
    :param y_real: A numpy array representing the real y-values without noise.
    :param output_folder: The folder where the plot image will be saved (default is "Results").
    :param filename: The name of the file without extension (default is "Data_and_Noisy_Data_Distribution").
    :param graph_title: The title of the graph (default is 'Synthetic Data Generation: True vs. Noisy Data').
    :param func: An optional callable to plot the true function if provided.
    :return: None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(10, 6))
    # Plot noisy data points
    plt.scatter(x, y_noisy, color="red", label="Noisy Data", alpha=0.6, marker="x")
    # Plot real data points
    plt.scatter(x, y_real, color="blue", label="True Data", alpha=0.6, marker="o")

    if func is not None:

        x_smooth = np.linspace(min(x), max(x), 100)
        y_smooth = func(x_smooth)
        plt.plot(
            x_smooth,
            y_smooth,
            color="green",
            label="True Function",
            linewidth=2,
            linestyle="--",
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(graph_title)
    plt.legend()

    plt.grid(True)

    plot_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_gaussian_bases(
    x, phi, D, output_folder="../Results", filename="Gaussian_Bases_Distribution"
):

    for d in range(D):
        plt.plot(x, phi[:, d], "-")

    plt.grid(True)

    plot_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(plot_path)
    plt.close()


def plot_model_fit(
    lr,
    x_values,
    y_values_noise,
    mu,
    num_bases,
    gaussian_func,
    func,
    precision=1000,
    data_range=(0, 20),
    output_folder="../Results",
    rescale_view=True,
    basis_func_name="Gaussian",
):
    """
    Plots the model's fit, noisy data, and Gaussian basis functions.


    :param lr (NonLinearRegression): Fitted model object.
    :param x_values: Original x values used for fitting.
    :param y_values_noise: Noisy y values.
    :param mu: Centers of the Gaussian bases.
    :param num_bases: Number of Gaussian basis functions.
    :param gaussian_func: Function to compute the Gaussian basis.
    :param func : True function to compare with the fit.
    :param precision: Number of data points for plotting (default: 1000).
    :param data_range : The range of x values (default: (0, 20)).
    :param output_folder: Folder where plots will be saved (default: "../Results").
    """

    # Recompute phi for the higher-precision x values (for plotting)
    x = np.linspace(data_range[0], data_range[1], precision)
    phi_plot = gaussian_func(x[:, None], mu[None, :], 1)

    # Generate predictions for the new x values
    y_h = lr.predict(phi_plot)

    # Scatter plot for the noisy data
    plt.scatter(
        x_values,
        y_values_noise,
        color="red",
        label="Noisy Data",
        alpha=0.6,
        marker="x",
    )

    # Plot contributions of each Gaussian basis function
    for d in range(num_bases):
        plt.plot(
            x,
            lr.weights[d] * phi_plot[:, d],  # Contribution of each basis function
            "-",
            alpha=0.5,
        )

        # Plot the overall fit
    plt.plot(x, y_h, "g-", label="Our fit")

    true_function_values = func(x)
    # Plot the true function (without noise)
    plt.plot(x, func(x), "b-", label="True Function")

    if rescale_view:
        plt.ylim(
            bottom=np.min(true_function_values) * 1.2,
            top=np.max(true_function_values) * 1.2,
        )

    # Add title and legend
    plt.title(f"Fitting with {num_bases} Gaussian Bases")
    plt.legend()

    os.makedirs(output_folder, exist_ok=True)

    if rescale_view:
        # Save the plot
        plot_path = os.path.join(
            output_folder,
            f"Scaled Fitting with {num_bases} {basis_func_name} Bases.png",
        )
    else:
        plot_path = os.path.join(
            output_folder, f"Fitting with {num_bases} {basis_func_name} Bases.png"
        )
    plt.savefig(plot_path)
    plt.close()


def plot_sse(
    num_bases_range,
    sse_list,
    output_folder="../Results",
    filename="SSE",
    title="Sum of Squared Errors vs. Number of Bases",
    log_scale=False,
):
    """
    Plot the sum of squared errors
    :param num_bases_range: number of Gaussian basis functions.
    :param sse_list: list of sum of squared errors vs. number of bases.
    :param output_folder: The folder where the plot will be saved (default is "../Results").
    :param filename: The name of the file without extension (default is "SSE").
    :param title: Title of the graph (default is "Sum of Squared Errors vs. Number of Bases").
    :param log_scale: If True, y is logarithmic scale.
    :return:
    """

    plt.figure(figsize=(10, 6))
    plt.plot(num_bases_range, sse_list, marker="o", label="SSE", color="blue")
    plt.title(title)
    plt.xlabel("Number of Bases")
    plt.ylabel("Sum of Squared Errors")
    plt.xticks(num_bases_range)
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


def plot_average_sse(
    num_bases_range,
    sse_list,
    output_folder="../Results",
    filename="Average SSE",
    title="Average Sum of Squared Errors vs. Number of Bases",
    log_scale=False,
):
    """
    Plot the average sum of squared errors
    :param num_bases_range: number of Gaussian basis functions.
    :param sse_list: list of sum of squared errors vs. number of bases.
    :param output_folder: Folder where plots will be saved (default: "../Results").
    :param filename: Name of the file to save the plots (default: "Average SSE").
    :param title: Title of the graph (default: "Average Sum of Squared Errors vs. Number of Bases").
    :param log_scale: If True, y is set to log scale
    :return: Nothing
    """
    """Plot the sum of squared errors"""
    plt.figure(figsize=(10, 6))
    plt.plot(num_bases_range, sse_list, marker="o", label="Average SSE", color="blue")
    plt.title(title)
    plt.xlabel("Number of Bases")
    plt.ylabel("Sum of Squared Errors")
    plt.xticks(num_bases_range)
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(output_folder, filename)
    plt.savefig(plot_path)
    plt.close()


def plot_average_fitted_models(
    x,
    all_fitted_models,
    func,
    num_bases,
    output_folder="../Results",
    rescale_view=False,
    basis_name="Gaussian",
):
    """
    Plot the average of fitted models along with the individual models and the true function.
    :param x: linspace of x values for plotting
    :param all_fitted_models: All fitted models in an array.
    :param func: True function to compare with the fit.
    :param num_bases: number of Gaussian basis functions.
    :param output_folder: Folder where plots will be saved (default: "../Results").
    :param rescale_view: Rescale the view of the fitted models before plotting.
    :return: Nothing
    """
    avg_fitted_model = np.mean(all_fitted_models, axis=0)

    for d in range(len(all_fitted_models)):
        if d == 0:
            plt.plot(
                x, all_fitted_models[d], color="#90EE90", label="All Fitted Models"
            )  # Add label only once
        else:
            plt.plot(
                x, all_fitted_models[d], color="#90EE90"
            )  # No label for subsequent lines

    true_function_values = func(x)
    plt.plot(x, avg_fitted_model, "r", label="Fitted Model")
    plt.plot(x, true_function_values, "b", label="Actual Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Fitted Models and Bias-Variance Visualization ({basis_name} Bases={num_bases})"
    )
    plt.legend()

    if rescale_view:
        plt.ylim(
            bottom=np.min(true_function_values) * 1.2,
            top=np.max(true_function_values) * 1.2,
        )

    plot_path = os.path.join(
        output_folder,
        f"Fitted_Models_and_Bias-Variance_Visualization_({basis_name} Bases={num_bases})",
    )
    plt.savefig(plot_path)
    plt.close()
