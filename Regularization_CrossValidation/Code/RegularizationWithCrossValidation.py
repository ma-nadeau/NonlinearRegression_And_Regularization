from Helpers import *
from LinearRegressionRegularization import *


def evaluate_models(x, y, x_rest, y_rest, n_rest, x_test, y_test, lambda_values, l2_reg=False):
    # For storing results
    train_errors = np.zeros(len(lambda_values))
    val_errors = np.zeros(len(lambda_values))
    test_errors = np.zeros(len(lambda_values))
    bias_squared_values = np.zeros(len(lambda_values))
    variance_values = np.zeros(len(lambda_values))
    bias_squared_and_variance_values = np.zeros(len(lambda_values))
    all_error_values = np.zeros(len(lambda_values))

    # Iterate over lambda values
    for idx, lambda_val in enumerate(lambda_values):
        fold_train_errors = []
        fold_val_errors = []
        fold_test_errors = []
        test_predictions = []  # Store each fold's test predictions for current lambda_val

        optimizer = GradientDescent(learning_rate=0.01, max_iters=1e4, epsilon=1e-8)
        if l2_reg:
            model = LinearRegression(l2_reg=lambda_val)
        else:
            model = LinearRegression(l1_reg=lambda_val)
        model.fit(x, y, optimizer)

        # Perform cross-validation
        for tr_indices, val_indices in cross_validate(n_rest):
            x_train, y_train = x_rest[tr_indices], y_rest[tr_indices]
            x_val, y_val = x_rest[val_indices], y_rest[val_indices]

            # Create Gaussian-transformed features
            x_train_gauss = create_gaussian_features(x_train, 70)
            x_val_gauss = create_gaussian_features(x_val, 70)
            x_test_gauss = create_gaussian_features(x_test, 70)

            # Initialize Gradient Descent Optimizer
            optimizer = GradientDescent(learning_rate=0.01, max_iters=1e4, epsilon=1e-8)

            # Initialize Linear Regression model with L2 (Ridge) or L1 (Lasso)
            if l2_reg:
                model = LinearRegression(l2_reg=lambda_val)  # Use l2_reg for Ridge
            else:
                model = LinearRegression(l1_reg=lambda_val)  # Use l1_reg
            model.fit(x_train_gauss, y_train, optimizer)

            # Predictions and error
            y_train_pred = model.predict(x_train_gauss)
            y_val_pred = model.predict(x_val_gauss)
            y_test_pred = model.predict(x_test_gauss)
            test_predictions.append(y_test_pred)  # Save the predictions for bias-variance calculation

            test_errors[idx] = np.mean((y_test_pred - y_test) ** 2)

            # Compute Mean Squared Error for training and validation
            train_mse = np.mean((y_train_pred - y_train) ** 2)
            val_mse = np.mean((y_val_pred - y_val) ** 2)
            test_mse = np.mean((y_test_pred - y_test) ** 2)

            fold_train_errors.append(train_mse)
            fold_val_errors.append(val_mse)
            fold_test_errors.append(test_mse)

        # Average over folds
        train_errors[idx] = np.mean(fold_train_errors)
        val_errors[idx] = np.mean(fold_val_errors)
        test_errors[idx] = np.mean(fold_test_errors)

        current_bias = []
        current_var = []
        current_bias_var = []
        current_all = []

        for i, predictions in enumerate(np.array(test_predictions)):
            predictions = np.array(predictions)  # Shape: (num_folds, num_test_samples)
            mean_predictions = np.mean(predictions, axis=0)

            # Bias^2
            bias_squared = np.mean((mean_predictions - y_test) ** 2)
            current_bias.append(bias_squared)

            # Variance
            variance = np.mean(np.var(predictions, axis=0, ddof=1))
            current_var.append(variance)

            current_bias_var.append(bias_squared+variance)

            # Noise estimation
            noise = np.mean((y_test - mean_predictions) ** 2) - variance - bias_squared
            current_all.append(bias_squared + variance + noise)
        bias_squared_and_variance_values[idx] = np.mean(current_bias_var)
        bias_squared_values[idx] = np.mean(current_bias)
        variance_values[idx] = np.mean(current_var)
        all_error_values[idx] = np.mean(current_all)

    return (train_errors, val_errors, test_errors, bias_squared_values, variance_values,
            bias_squared_and_variance_values, all_error_values)


def initialize_parameters(num_instances=20, num_datasets=50):
    n_test = num_instances // 10
    lambdaa = np.logspace(-4, 2, 25)

    cumulative_train_errors_l1 = np.zeros(len(lambdaa))
    cumulative_val_errors_l1 = np.zeros(len(lambdaa))
    cumulative_test_errors_l1 = np.zeros(len(lambdaa))

    cumulative_train_errors_l2 = np.zeros(len(lambdaa))
    cumulative_val_errors_l2 = np.zeros(len(lambdaa))
    cumulative_test_errors_l2 = np.zeros(len(lambdaa))

    cumulative_bias_l1 = np.zeros(len(lambdaa))
    cumulative_variance_l1 = np.zeros(len(lambdaa))
    cumulative_bias_variance_l1 =np.zeros(len(lambdaa))
    cumulative_all_l1 = np.zeros(len(lambdaa))

    cumulative_bias_l2 = np.zeros(len(lambdaa))
    cumulative_variance_l2 = np.zeros(len(lambdaa))
    cumulative_bias_variance_l2 = np.zeros(len(lambdaa))
    cumulative_all_l2 = np.zeros(len(lambdaa))

    return (num_datasets, n_test, lambdaa, cumulative_train_errors_l1, cumulative_val_errors_l1,
            cumulative_test_errors_l1, cumulative_train_errors_l2, cumulative_val_errors_l2, cumulative_test_errors_l2
            ,cumulative_bias_l1, cumulative_variance_l1, cumulative_bias_variance_l1, cumulative_all_l1,
            cumulative_bias_l2, cumulative_variance_l2, cumulative_bias_variance_l2, cumulative_all_l2,
    )


def process_datasets(num_datasets, n_test, lambdaa, cumulative_train_errors_l1, cumulative_val_errors_l1,
                     cumulative_test_errors_l1, cumulative_train_errors_l2, cumulative_val_errors_l2,
                     cumulative_test_errors_l2, cumulative_bias_l1, cumulative_variance_l1, cumulative_bias_variance_l1,
                     cumulative_all_l1, cumulative_bias_l2, cumulative_variance_l2, cumulative_bias_variance_l2, cumulative_all_l2, num_instances=20):

    for i in range(num_datasets):
        np.random.seed(42)
        x, y = generate_data()
        indices = np.random.permutation(num_instances)
        x_test, y_test = x[indices[:n_test]], y[indices[:n_test]]
        x_rest, y_rest = x[indices[n_test:]], y[indices[n_test:]]
        n_rest = num_instances - n_test

        # Evaluate L1 regularization
        train_e_l1, val_e_l1, test_e_l1, bias_l1, var_l1, bias_and_var_l1, all_l1 = evaluate_models(x, y, x_rest,
                                                                                                    y_rest, n_rest,
                                                                                                    x_test, y_test,
                                                                                                    lambdaa)

        # Evaluate L2 regularization
        train_e_l2, val_e_l2, test_e_l2, bias_l2, var_l2, bias_and_var_l2, all_l2 = evaluate_models(x, y, x_rest,
                                                                                                    y_rest, n_rest,
                                                                                                    x_test,y_test,
                                                                                                    lambdaa,
                                                                                                    l2_reg=True)

        cumulative_train_errors_l1 += train_e_l1
        cumulative_val_errors_l1 += val_e_l1
        cumulative_test_errors_l1 += test_e_l1

        cumulative_bias_l1 += bias_l1
        cumulative_variance_l1 += var_l1
        cumulative_bias_variance_l1 += bias_and_var_l1
        cumulative_all_l1 += all_l1

        cumulative_train_errors_l2 += train_e_l2
        cumulative_val_errors_l2 += val_e_l2
        cumulative_test_errors_l2 += test_e_l2

        cumulative_bias_l2 += bias_l2
        cumulative_variance_l2 += var_l2
        cumulative_bias_variance_l2 += bias_and_var_l2
        cumulative_all_l2 += all_l2

    return (cumulative_train_errors_l1, cumulative_val_errors_l1, cumulative_test_errors_l1, cumulative_train_errors_l2,
            cumulative_val_errors_l2, cumulative_test_errors_l2,
            cumulative_bias_l1,
            cumulative_variance_l1,
            cumulative_bias_variance_l1,
            cumulative_all_l1,
            cumulative_bias_l2,
            cumulative_variance_l2,
            cumulative_bias_variance_l2,
            cumulative_all_l2
            )


def calculate_average_errors(cumulative_train_errors_l1, cumulative_val_errors_l1, cumulative_test_errors_l1,
                             cumulative_train_errors_l2, cumulative_val_errors_l2, cumulative_test_errors_l2,
                             cumulative_bias_l1, cumulative_variance_l1, cumulative_bias_variance_l1, cumulative_all_l1,
                             cumulative_bias_l2, cumulative_variance_l2, cumulative_bias_variance_l2, cumulative_all_l2,
                             num_datasets):
    avg_train_errors_l1 = cumulative_train_errors_l1 / num_datasets
    avg_val_errors_l1 = cumulative_val_errors_l1 / num_datasets
    avg_test_errors_l1 = cumulative_test_errors_l1 / num_datasets

    avg_train_errors_l2 = cumulative_train_errors_l2 / num_datasets
    avg_val_errors_l2 = cumulative_val_errors_l2 / num_datasets
    avg_test_errors_l2 = cumulative_test_errors_l2 / num_datasets

    avg_bias_l1 = cumulative_bias_l1 / num_datasets
    avg_bias_l2 = cumulative_bias_l2 / num_datasets
    avg_all_l1 = cumulative_all_l1 / num_datasets
    avg_all_l2 = cumulative_all_l2 / num_datasets
    avg_variance_l1 = cumulative_variance_l1 / num_datasets
    avg_variance_l2 = cumulative_variance_l2 / num_datasets
    avg_bias_variance_l1 = cumulative_bias_variance_l1 / num_datasets
    avg_bias_variance_l2 = cumulative_bias_variance_l2 / num_datasets

    return (avg_train_errors_l1, avg_val_errors_l1, avg_test_errors_l1, avg_train_errors_l2, avg_val_errors_l2,
            avg_test_errors_l2, avg_bias_l1, avg_bias_l2, avg_all_l1, avg_all_l2, avg_variance_l1, avg_variance_l2,
            avg_bias_variance_l1, avg_bias_variance_l2)


def main():
    # Initialize parameters and errors
    (num_datasets, n_test, lambdaa, cumulative_train_errors_l1, cumulative_val_errors_l1, cumulative_test_errors_l1,
     cumulative_train_errors_l2, cumulative_val_errors_l2, cumulative_test_errors_l2,cumulative_bias_l1,
     cumulative_variance_l1, cumulative_bias_variance_l1, cumulative_all_l1, cumulative_bias_l2,
     cumulative_variance_l2, cumulative_bias_variance_l2, cumulative_all_l2) = initialize_parameters()

    # Process datasets
    (cumulative_train_errors_l1, cumulative_val_errors_l1, cumulative_test_errors_l1, cumulative_train_errors_l2,
     cumulative_val_errors_l2, cumulative_test_errors_l2, cumulative_bias_l1, cumulative_variance_l1,
     cumulative_bias_variance_l1, cumulative_all_l1, cumulative_bias_l2, cumulative_variance_l2,
     cumulative_bias_variance_l2, cumulative_all_l2,
     ) = (
        process_datasets(num_datasets, n_test, lambdaa, cumulative_train_errors_l1, cumulative_val_errors_l1,
                         cumulative_test_errors_l1, cumulative_train_errors_l2, cumulative_val_errors_l2,
                         cumulative_test_errors_l2, cumulative_bias_l1, cumulative_variance_l1,
                         cumulative_bias_variance_l1, cumulative_all_l1, cumulative_bias_l2, cumulative_variance_l2,
                         cumulative_bias_variance_l2, cumulative_all_l2))

    # Calculate average errors
    (avg_train_errors_l1, avg_val_errors_l1, avg_test_errors_l1, avg_train_errors_l2, avg_val_errors_l2,
     avg_test_errors_l2, avg_bias_l1, avg_bias_l2, avg_all_l1, avg_all_l2, avg_variance_l1, avg_variance_l2,
     avg_bias_variance_l1, avg_bias_variance_l2) = calculate_average_errors(
        cumulative_train_errors_l1, cumulative_val_errors_l1, cumulative_test_errors_l1, cumulative_train_errors_l2,
        cumulative_val_errors_l2, cumulative_test_errors_l2, cumulative_bias_l1, cumulative_variance_l1,
        cumulative_bias_variance_l1, cumulative_all_l1, cumulative_bias_l2, cumulative_variance_l2,
        cumulative_bias_variance_l2, cumulative_all_l2, num_datasets)

    # Save results to CSV
    save_results_to_csv(lambdaa, avg_train_errors_l1, avg_val_errors_l1, avg_train_errors_l2, avg_val_errors_l2)

    # Plot L1 regularization results
    plot_errors(lambdaa, avg_train_errors_l1, avg_test_errors_l1, avg_val_errors_l1,
                'Train vs Test vs Validation Error (L1 Regularization)', "../Results/l1_regularization_plot.png")

    # Plot L2 regularization results
    plot_errors(lambdaa, avg_train_errors_l2, avg_test_errors_l2, avg_val_errors_l2,
                'Train vs Test vs Validation Error (L2 Regularization)', "../Results/l2_regularization_plot.png")

    plot_bias_variance_decomposition(lambdaa, avg_bias_l1, avg_variance_l1,avg_bias_variance_l1, avg_all_l1,
                                     'Bias-Variance Decomposition (L1 Regularization)',
                                     "../Results/l1_decomposition_plot.png")

    plot_bias_variance_decomposition(lambdaa, avg_bias_l2, avg_variance_l2, avg_bias_variance_l2, avg_all_l2,
                                     'Bias-Variance Decomposition (L2 Regularization)',
                                     "../Results/l2_decomposition_plot.png")


if __name__ == "__main__":
    main()
