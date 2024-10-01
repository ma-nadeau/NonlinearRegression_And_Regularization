import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from typing import Callable, Tuple, Optional


def plot_real_data_and_noisy_data(
    x: np.array,
    y_noisy: np.array,
    y_real: np.array,
    output_folder: str = "Results",
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
