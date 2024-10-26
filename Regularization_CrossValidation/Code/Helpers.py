import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def generate_data(n_points=20):
    x = np.linspace(0, 20, n_points)
    epsilon = np.random.normal(0, 1, n_points)
    y = np.sin(np.sqrt(x)) + np.cos(x) + np.sin(x) + epsilon
    return x, y


def gaussian_basis_function(x, mu, sigma=1):
    return np.exp(- ((x - mu) / sigma) ** 2)


def create_gaussian_features(x, n_bases=70, sigma=1):
    mus = np.linspace(np.min(x), np.max(x), n_bases)
    features = np.column_stack([gaussian_basis_function(x, mu, sigma) for mu in mus])
    return features


def cross_validate(n, n_folds=10):
    # Get the number of data samples in each split
    n_val = n // n_folds
    for f in range(n_folds):
        tr_indices = []
        # Get the validation indexes
        val_indices = list(range(f * n_val, (f + 1) * n_val))
        # Get the train indexes
        if f > 0:
            tr_indices = list(range(f * n_val))
        if f < n_folds - 1:
            tr_indices = tr_indices + list(range((f + 1) * n_val, n))
        # The yield statement suspends function’s execution and sends a value back to the caller
        # but retains enough state information to enable function to resume where it is left off
        yield tr_indices, val_indices


def plot_errors(lambdaa, avg_train_errors, avg_test_errors, avg_val_errors, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(lambdaa, avg_train_errors, label='Train Error', marker='o')
    plt.plot(lambdaa, avg_test_errors, label='Test Error', marker='o')
    plt.plot(lambdaa, avg_val_errors, label='Validation Error', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Error (MSE)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def save_results_to_csv(lambdaa, avg_train_errors_l1, avg_val_errors_l1, avg_train_errors_l2, avg_val_errors_l2):
    comparison_data = {
        'Lambda': lambdaa,
        'Train Error (L1)': avg_train_errors_l1,
        'Validation Error (L1)': avg_val_errors_l1,
        'Train Error (L2)': avg_train_errors_l2,
        'Validation Error (L2)': avg_val_errors_l2
    }

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("../Results/train_and_validation_error.csv", index=False)


def plot_bias_variance_decomposition(lambdaa, avg_bias, avg_variance, avg_bias_variance, avg_all,
                                     title, filename):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis (Variance, Bias2 + Variance, Bias2 + Variance + Noise)
    ax1.plot(lambdaa, avg_variance, label='Variance', marker='o')
    ax1.plot(lambdaa, avg_bias_variance, label='Bias2 + Variance', marker='o')
    ax1.plot(lambdaa, avg_all, label='Bias2 + Variance + Noise', marker='o')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Lambda')
    ax1.set_ylabel('Test Error')
    ax1.grid(True)

    # Create a second y-axis for Bias^2
    ax2 = ax1.twinx()
    ax2.plot(lambdaa, avg_bias, label='Bias2', marker='o', color='r')
    ax2.set_yscale('log')  # Ensure the scale is consistent
    ax2.set_ylabel('Bias2')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(title)
    plt.savefig(filename)
    plt.show()
