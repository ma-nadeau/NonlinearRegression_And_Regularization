import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from typing import Callable, Tuple, Optional
import itertools


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

    # Plot the overall fit
    plt.plot(x, y_h, "g-", label="Our fit")

    # Plot the true function (without noise)
    plt.plot(x, func(x), "b-", label="True Function")

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

    # Add title and legend
    plt.title(f"Fitting with {num_bases} Gaussian Bases")
    plt.legend()

    os.makedirs(output_folder, exist_ok=True)

    # Save the plot
    plot_path = os.path.join(
        output_folder, f"Fitting with {num_bases} Gaussian Bases.png"
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
    plt.plot(x, avg_fitted_model, "r", label="Fitted Model")

    for d in range(len(all_fitted_models)):
        if d == 0:
            plt.plot(
                x, all_fitted_models[d], "g", label="All Fitted Models"
            )  # Add label only once
        else:
            plt.plot(x, all_fitted_models[d], "g")  # No label for subsequent lines

    true_function_values = func(x)
    plt.plot(x, true_function_values, "b", label="Actual Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Fitted Models and Bias-Variance Visualization (Bases={num_bases})")
    plt.legend()

    if rescale_view:
        plt.ylim(
            bottom=np.min(true_function_values) * 1.2,
            top=np.max(true_function_values) * 1.2,
        )

    plot_path = os.path.join(
        output_folder,
        f"Fitted_Models_and_Bias-Variance_Visualization_(Bases={num_bases})",
    )
    plt.savefig(plot_path)
    plt.close()


def plot_contour(f, x1bound, x2bound, resolution, ax):
    x1range = np.linspace(x1bound[0], x1bound[1], resolution)
    x2range = np.linspace(x2bound[0], x2bound[1], resolution)
    xg, yg = np.meshgrid(x1range, x2range)
    zg = np.zeros_like(xg)
    for i,j in itertools.product(range(resolution), range(resolution)):
        zg[i,j] = f([xg[i,j], yg[i,j]])
    ax.contour(xg, yg, zg, 100)
    return ax

def plot_cost_func_contour(x,y):
    cost = lambda w: .5 * np.mean((w[0] + w[1] * x - y) ** 2)
    l2_penalty = lambda w: np.dot(w, w) / 2
    l1_penalty = lambda w: np.sum(np.abs(w))
    strength = [5]
    fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True, figsize=(15, 5))
    plot_contour(cost, [-20, 20], [-20, 20], 50, axes[0])
    axes[0].set_title(r'cost function $J(w)$')
    plot_contour(l2_penalty, [-20, 20], [-20, 20], 50, axes[1])
    axes[1].set_title(r'L2 reg. $||w||_2^2$')
    for i in range(len(strength)):
        cost_plus_l2 = lambda w: cost(w) + strength[i] * l2_penalty(w)
        plot_contour(cost_plus_l2, [-20, 20], [-20, 20], 50, axes[i + 2])
        axes[i + 2].set_title(r'L2 reg. cost $J(w) + ' + str(strength[i]) + ' ||w||_2^2$')
    plot_path = os.path.join(
        "../Results",
        f"Cost_L2_strength_5",
    )
    plt.savefig(plot_path)
    plt.close()
    strength = [10,15,20]
    fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True, figsize=(15, 5))
    for i in range(len(strength)):
        cost_plus_l2 = lambda w: cost(w) + strength[i] * l2_penalty(w)
        plot_contour(cost_plus_l2, [-20, 20], [-20, 20], 50, axes[i])
        axes[i].set_title(r'L2 reg. cost $J(w) + ' + str(strength[i]) + ' ||w||_2^2$')
    plot_path = os.path.join(
        "../Results",
        f"Cost_L2_strength_10_15_20",
    )
    plt.savefig(plot_path)
    plt.close()
    strength = [5]
    fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True, figsize=(15, 5))
    plot_contour(cost, [-20, 20], [-20, 20], 50, axes[0])
    axes[0].set_title(r'cost function $J(w)$')
    plot_contour(l1_penalty, [-20, 20], [-20, 20], 50, axes[1])
    axes[1].set_title(r'L1 reg. $||w||_2^2$')
    for i in range(len(strength)):
        cost_plus_l1 = lambda w: cost(w) + strength[i] * l1_penalty(w)
        plot_contour(cost_plus_l1, [-20, 20], [-20, 20], 50, axes[i + 2])
        axes[i + 2].set_title(r'L1 reg. cost $J(w) + ' + str(strength[i]) + ' ||w||_2^2$')
    plot_path = os.path.join(
        "../Results",
        f"Cost_L1_strength_5",
    )
    plt.savefig(plot_path)
    plt.close()
    strength = [10, 15, 20]
    fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True, figsize=(15, 5))
    for i in range(len(strength)):
        cost_plus_l1 = lambda w: cost(w) + strength[i] * l1_penalty(w)
        plot_contour(cost_plus_l1, [-20, 20], [-20, 20], 50, axes[i])
        axes[i].set_title(r'L1 reg. cost $J(w) + ' + str(strength[i]) + ' ||w||_2^2$')
    plot_path = os.path.join(
        "../Results",
        f"Cost_L1_strength_10_15_20",
    )
    plt.savefig(plot_path)
    plt.close()

def plot_gradient_descent(current_cost,optimizer,axes,reg_coef,L2 = True):
    plot_contour(current_cost, [-20, 20], [-5, 5], 50, axes)
    w_hist = np.vstack(optimizer.w_history)  # T x 2
    axes.plot(w_hist[:, 1], w_hist[:, 0], '.r', alpha=.8)
    axes.plot(w_hist[:, 1], w_hist[:, 0], '-r', alpha=.3)
    axes.set_xlabel(r'$w_0$')
    axes.set_ylabel(r'$w_1$')
    axes.set_title(f' lambda = {reg_coef}')
    axes.set_xlim([-20, 20])
    axes.set_ylim([-5, 5])
    if L2:
        plot_path = os.path.join(
            "../Results",
            f"L2_Gradient_descent_lambda = {reg_coef}",
        )
        plt.savefig(plot_path)
        plt.close()
    else:
        plot_path = os.path.join(
            "../Results",
            f"L1_Gradient_descent_lambda = {reg_coef}",
        )
        plt.savefig(plot_path)
        plt.close()
