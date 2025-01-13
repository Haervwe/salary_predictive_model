from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Tuple
import numpy as np
from sklearn.utils import resample

def evaluate_model(normalized_X_test:pd.DataFrame,y_test:pd.Series,normalized_X_train:pd.DataFrame,y_train:pd.Series,model)->None:
 
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
    plt.show()
    plt.close()  # Close the figure to free up memory

    # Plot residuals vs. predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_best_rf, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted Salaries')
    plt.xlabel('Predicted Salaries')
    plt.ylabel('Residuals')
    plt.show()
    plt.close()
    
    # Initialize the explainer with the Random Forest model
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(normalized_X_test)

    # Plot the summary plot
    shap.summary_plot(shap_values, normalized_X_test, plot_type='bar')
    
    train_data = normalized_X_train.copy()
    train_data['Salary'] = y_train

    # Pairplot
    sns.pairplot(train_data)
    plt.show()

    # Correlation heatmap
    corr_matrix = train_data.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.show()
    
    
    # Calculate and display performance metrics after hyperparameter tuning
    print("Random Forest Regressor Performance After Hyperparameter Tuning:")
    print(f"Mean Squared Error (MSE): {mse_best_rf:.2f}")
    print(f"Mean Absolute Error (MAE): {mae_best_rf:.2f}")
    print(f"R-squared Score (R²): {r2_best_rf:.2f}")
    
    
    
def evaluate_NN_model(normalized_X_test:pd.DataFrame, y_test:pd.Series,model_nn):
    # Predict using a nn model.
    y_pred_nn = model_nn.predict(normalized_X_test).flatten()
    
    # Calculate residuals
    residuals_nn = y_test - y_pred_nn

    # Plot residuals histogram
    plt.figure()
    sns.histplot(residuals_nn, kde=True)
    plt.title('Distribution of Residuals (Neural Network)')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig('plots/nn_residuals_histogram.png', bbox_inches='tight')
    plt.close()

    # Plot residuals vs. predicted values
    plt.figure()
    plt.scatter(y_pred_nn, residuals_nn)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted Salaries (Neural Network)')
    plt.xlabel('Predicted Salaries')
    plt.ylabel('Residuals')
    plt.savefig('plots/nn_residuals_vs_predicted.png', bbox_inches='tight')
    plt.close()
    
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)
    print("Neural Network Performance:")
    print(f"MSE: {mse_nn:.2f}")
    print(f"R-squared Score (R²): {r2_nn:.2f}")
    

def calculate_metric_with_ci(model, X_test:pd.DataFrame, y_test:pd.DataFrame, metric_func, n_bootstraps:int=1000, alpha:float=0.90, random_state:int=42)->Tuple[float, float, float]:
    """
    Calculates the metric and its confidence interval using bootstrapping.

    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: True target values
    - metric_func: Function to compute the metric (e.g., mean_squared_error)
    - n_bootstraps: Number of bootstrap samples
    - alpha: Confidence level (e.g., 0.95 for 95% confidence interval)
    - random_state: Seed for reproducibility

    Returns:
    - metric_mean: Mean value of the metric across bootstrap samples
    - ci_lower: Lower bound of the confidence interval
    - ci_upper: Upper bound of the confidence interval
    """
    np.random.seed(random_state)

    metrics = []
    n_samples = len(y_test)
    indices = np.arange(n_samples)

    for _ in range(n_bootstraps):
        # Resample indices with replacement
        sample_indices = resample(indices, replace=True, n_samples=n_samples)
        # Get bootstrap samples
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]
        # Predict and compute metric
        y_pred_sample = model.predict(X_sample)
        metric_value = metric_func(y_sample, y_pred_sample)
        metrics.append(metric_value)

    metric_mean = np.mean(metrics)
    lower_percentile = ((1.0 - alpha) / 2.0) * 100
    upper_percentile = (alpha + ((1.0 - alpha) / 2.0)) * 100
    ci_lower = np.percentile(metrics, lower_percentile)
    ci_upper = np.percentile(metrics, upper_percentile)

    return metric_mean, ci_lower, ci_upper


def calculate_metrics( X_test:pd.DataFrame, y_test:pd.DataFrame, model)->None:
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

    # For R², we need to adjust the function slightly since r2_score can sometimes return values outside [-1, 1] when bootstrapping
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
        r2 = adjusted_r2_score(y_sample, y_pred_sample)
        r2_scores.append(r2)

    r2_mean = np.mean(r2_scores)
    alpha = 0.95
    lower_percentile = ((1.0 - alpha) / 2.0) * 100
    upper_percentile = (alpha + ((1.0 - alpha) / 2.0)) * 100
    r2_ci_lower = np.percentile(r2_scores, lower_percentile)
    r2_ci_upper = np.percentile(r2_scores, upper_percentile)
    print("Random Forest Regressor Performance with Confidence Intervals:")
    print(f"Mean Squared Error (MSE): {mse_mean:.2f} (95% CI: [{mse_ci_lower:.2f}, {mse_ci_upper:.2f}])")
    print(f"Mean Absolute Error (MAE): {mae_mean:.2f} (95% CI: [{mae_ci_lower:.2f}, {mae_ci_upper:.2f}])")
    print(f"R-squared Score (R²): {r2_mean:.2f} (95% CI: [{r2_ci_lower:.2f}, {r2_ci_upper:.2f}])")
    