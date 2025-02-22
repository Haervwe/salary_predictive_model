import logging
logging.getLogger('shap').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
from typing import Tuple, Dict
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def evaluate_model(normalized_X_test: pd.DataFrame, y_test: pd.Series, 
                  normalized_X_train: pd.DataFrame, y_train: pd.Series, 
                  model) -> Dict:
    """Original evaluate_model function with added return value"""
    y_pred_best_rf = model.predict(normalized_X_test)
    mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
    mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
    r2_best_rf = r2_score(y_test, y_pred_best_rf)

    residuals = y_test - y_pred_best_rf

    # Plot residuals histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig('plots/distribution_of_residuals.png', bbox_inches='tight')
    print("Distribution of Residuals plot saved to 'plots/distribution_of_residuals.png'")
    plt.close()

    # Plot residuals vs. predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_best_rf, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted Salaries')
    plt.xlabel('Predicted Salaries')
    plt.ylabel('Residuals')
    plt.savefig('plots/residuals_vs_predicted_salaries.png', bbox_inches='tight')
    print("Residuals vs. Predicted Salaries plot saved to 'plots/residuals_vs_predicted_salaries.png'")
    plt.close()
    
    # Initialize the explainer with the Random Forest model
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(normalized_X_test)

    # Plot the summary plot
    shap.summary_plot(shap_values, normalized_X_test, plot_type='bar', show=False)
    plt.savefig('plots/shap_summary_plot_rf.png', bbox_inches='tight')
    print("SHAP Summary Plot saved to 'plots/shap_summary_plot_rf.png'")
    plt.close()
    
    train_data = normalized_X_train.copy()
    train_data['Salary'] = y_train


    
    # Calculate and display performance metrics after hyperparameter tuning
    print("Random Forest Regressor Performance After Hyperparameter Tuning:")
    print(f"Mean Squared Error (MSE): {mse_best_rf:.2f}")
    print(f"Mean Absolute Error (MAE): {mae_best_rf:.2f}")
    print(f"R-squared Score (R²): {r2_best_rf:.2f}")
    
    return {
        'mse': mse_best_rf,
        'mae': mae_best_rf,
        'r2': r2_best_rf,
        'predictions': y_pred_best_rf,
        'residuals': residuals
    }

def evaluate_NN_model(normalized_X_test: pd.DataFrame, y_test: pd.Series,
                     normalized_X_train: pd.DataFrame, model_nn) -> Dict:
    """
    Evaluates the neural network model on the test data, calculates SHAP values,
    and plots the results.
    
    Parameters:
    - normalized_X_test: Normalized test features as a DataFrame
    - y_test: True target values for the test set
    - normalized_X_train: Normalized training features as a DataFrame
    - model_nn: Trained neural network model
    
    Returns:
    - Dictionary containing MSE, MAE, R-squared, predictions, and residuals
    """

    
    # Predict using the neural network model
    y_pred_nn = model_nn.predict(normalized_X_test, verbose=0).flatten()

    
    # Calculate residuals
    residuals_nn = y_test - y_pred_nn
    
    # Plot residuals histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals_nn, kde=True)
    plt.title('Distribution of Residuals (Neural Network)')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig('plots/distribution_of_residuals_nn.png', bbox_inches='tight')
    print("Distribution of Residuals (Neural Network) plot saved to 'plots/distribution_of_residuals_nn.png'")
    plt.close()
    
    # Plot residuals vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_nn, residuals_nn)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted Salaries (Neural Network)')
    plt.xlabel('Predicted Salaries')
    plt.ylabel('Residuals')
    plt.savefig('plots/residuals_vs_predicted_salaries_nn.png', bbox_inches='tight')
    print("Residuals vs. Predicted Salaries (Neural Network) plot saved to 'plots/residuals_vs_predicted_salaries_nn.png'")
    plt.close()

    

    
    # Create wrapper function for the model predictions
    def model_predict(x):
        return model_nn.predict(x, verbose=0).flatten() 
    
    try:
        background_size = min(100, len(normalized_X_train))
        background_indices = np.random.choice(len(normalized_X_train), size=background_size, replace=False)
        background_data = normalized_X_train.iloc[background_indices].values  # Convert to NumPy array

        explainer = shap.KernelExplainer(model_predict, background_data) #No feature names needed

        # Use a smaller sample for SHAP if the test set is large
        nsamples = min(100, len(normalized_X_test)) #nsamples = 'auto' sometimes causes errors
        sample_indices = np.random.choice(len(normalized_X_test), nsamples, replace=False)

        test_samples = normalized_X_test.iloc[sample_indices].values
        shap_values = explainer.shap_values(test_samples, nsamples=min(50, nsamples)) #nsamples parameter


        shap.summary_plot(shap_values, normalized_X_test.iloc[sample_indices], plot_type="bar", show=False)
        plt.savefig('plots/shap_summary_plot_nn.png', bbox_inches='tight')
        print("SHAP Summary Plot (Neural Network) saved to 'plots/shap_summary_plot_nn.png'")
        plt.close()
    except Exception as e:
        print(f"SHAP analysis failed with error: {str(e)}")
    
    
    # Calculate performance metrics
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)
    
    print("\nNeural Network Performance:")
    print(f"MSE: {mse_nn:.2f}")
    print(f"MAE: {mae_nn:.2f}")
    print(f"R-squared Score (R²): {r2_nn:.2f}")
    
    return {
        'mse': mse_nn,
        'mae': mae_nn,
        'r2': r2_nn,
        'predictions': y_pred_nn,
        'residuals': residuals_nn
    }
    
    
    
    
def calculate_metric_with_ci(model, X_test: pd.DataFrame, y_test: pd.DataFrame, 
                           metric_func, n_bootstraps: int = 1000, 
                           alpha: float = 0.90, random_state: int = 42) -> Tuple[float, float, float]:
    """Original calculate_metric_with_ci function"""
    np.random.seed(random_state)

    metrics = []
    n_samples = len(y_test)
    indices = np.arange(n_samples)

    for _ in range(n_bootstraps):
        sample_indices = resample(indices, replace=True, n_samples=n_samples)
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]
        y_pred_sample = model.predict(X_sample)
        if isinstance(y_pred_sample, np.ndarray) and len(y_pred_sample.shape) > 1:
            y_pred_sample = y_pred_sample.flatten()
        metric_value = metric_func(y_sample, y_pred_sample)
        metrics.append(metric_value)

    metric_mean = np.mean(metrics)
    lower_percentile = ((1.0 - alpha) / 2.0) * 100
    upper_percentile = (alpha + ((1.0 - alpha) / 2.0)) * 100
    ci_lower = np.percentile(metrics, lower_percentile)
    ci_upper = np.percentile(metrics, upper_percentile)

    return metric_mean, ci_lower, ci_upper

def calculate_metrics(X_test: pd.DataFrame, y_test: pd.DataFrame, model) -> Dict:
    """Original calculate_metrics function with added return value"""
    # Calculate MSE with confidence interval
    mse_mean, mse_ci_lower, mse_ci_upper = calculate_metric_with_ci(
        model=model,
        X_test=X_test,
        y_test=y_test,
        metric_func=mean_squared_error
    )

    # Calculate MAE with confidence interval
    mae_mean, mae_ci_lower, mae_ci_upper = calculate_metric_with_ci(
        model=model,
        X_test=X_test,
        y_test=y_test,
        metric_func=mean_absolute_error
    )

    def adjusted_r2_score(y_true, y_pred):
        return r2_score(y_true, y_pred)

    r2_scores = []
    n_bootstraps = 1000
    n_samples = len(y_test)
    indices = np.arange(n_samples)

    np.random.seed(42)
    for _ in range(n_bootstraps):
        sample_indices = resample(indices, replace=True, n_samples=n_samples)
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]
        y_pred_sample = model.predict(X_sample)
        if isinstance(y_pred_sample, np.ndarray) and len(y_pred_sample.shape) > 1:
            y_pred_sample = y_pred_sample.flatten()
        r2 = adjusted_r2_score(y_sample, y_pred_sample)
        r2_scores.append(r2)

    r2_mean = np.mean(r2_scores)
    alpha = 0.95
    lower_percentile = ((1.0 - alpha) / 2.0) * 100
    upper_percentile = (alpha + ((1.0 - alpha) / 2.0)) * 100
    r2_ci_lower = np.percentile(r2_scores, lower_percentile)
    r2_ci_upper = np.percentile(r2_scores, upper_percentile)
    
    print("Model Performance with Confidence Intervals:")
    print(f"Mean Squared Error (MSE): {mse_mean:.2f} (95% CI: [{mse_ci_lower:.2f}, {mse_ci_upper:.2f}])")
    print(f"Mean Absolute Error (MAE): {mae_mean:.2f} (95% CI: [{mae_ci_lower:.2f}, {mae_ci_upper:.2f}])")
    print(f"R-squared Score (R²): {r2_mean:.2f} (95% CI: [{r2_ci_lower:.2f}, {r2_ci_upper:.2f}])")
    
    return {
        'mse': {'mean': mse_mean, 'ci_lower': mse_ci_lower, 'ci_upper': mse_ci_upper},
        'mae': {'mean': mae_mean, 'ci_lower': mae_ci_lower, 'ci_upper': mae_ci_upper},
        'r2': {'mean': r2_mean, 'ci_lower': r2_ci_lower, 'ci_upper': r2_ci_upper}
    }