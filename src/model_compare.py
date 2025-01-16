import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample

def calculate_bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              n_bootstraps: int = 1000, alpha: float = 0.95) -> Dict:
    """
    Calculate metrics with confidence intervals using bootstrapping.
    """
    n_samples = len(y_true)
    bootstrap_metrics = {
        'mse': [],
        'mae': [],
        'r2': []
    }
    
    for _ in range(n_bootstraps):
        # Bootstrap sample indices
        indices = np.random.randint(0, n_samples, n_samples)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        
        # Calculate metrics for this bootstrap sample
        bootstrap_metrics['mse'].append(mean_squared_error(sample_true, sample_pred))
        bootstrap_metrics['mae'].append(mean_absolute_error(sample_true, sample_pred))
        bootstrap_metrics['r2'].append(r2_score(sample_true, sample_pred))
    
    # Calculate confidence intervals
    results = {}
    for metric in bootstrap_metrics:
        values = np.array(bootstrap_metrics[metric])
        mean_val = np.mean(values)
        ci_lower = np.percentile(values, ((1 - alpha) / 2) * 100)
        ci_upper = np.percentile(values, (alpha + ((1 - alpha) / 2)) * 100)
        
        results[metric] = {
            'mean': mean_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return results

def compare_models(models_data: Dict[str, Tuple[object, pd.DataFrame, pd.Series]], 
                  y_test: pd.Series, n_bootstraps: int = 1000) -> Dict:
    """
    Compare multiple models' performance metrics with confidence intervals.
    
    Parameters:
    -----------
    models_data : Dict[str, Tuple[object, pd.DataFrame, pd.Series]]
        Dictionary where:
        - key: model name (str)
        - value: tuple containing (model, normalized_X_test, y_test)
    y_test : pd.Series
        True target values for comparison
    n_bootstraps : int
        Number of bootstrap samples for confidence interval calculation
        
    Returns:
    --------
    Dict
        Dictionary containing comparison metrics and confidence intervals
    """
    results = {}
    
    for name, (model, X_test_normalized, _) in models_data.items():
        # Get predictions
        y_pred = model.predict(X_test_normalized)
        if isinstance(y_pred, np.ndarray) and len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
            
        # Calculate metrics with confidence intervals
        metrics = calculate_bootstrap_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            n_bootstraps=n_bootstraps
        )
        
        results[name] = {
            'predictions': y_pred,
            'metrics': metrics
        }
    
    # Create visualizations with confidence intervals
    
    # 1. Metrics comparison with error bars
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metric_names = ['mse', 'mae', 'r2']
    metric_labels = ['Mean Squared Error', 'Mean Absolute Error', 'R-squared']
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        means = [results[model]['metrics'][metric]['mean'] for model in results]
        errors = [
            [results[model]['metrics'][metric]['mean'] - results[model]['metrics'][metric]['ci_lower'] for model in results],
            [results[model]['metrics'][metric]['ci_upper'] - results[model]['metrics'][metric]['mean'] for model in results]
        ]
        
        axes[idx].bar(results.keys(), means, yerr=errors, capsize=5)
        axes[idx].set_title(label)
        axes[idx].set_xticklabels(results.keys(), rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(means):
            axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    file_name = "./plots/error_model_comparison.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
    # 2. Predictions vs Actual scatter plot with confidence bands
    plt.figure(figsize=(15, 5))
    for i, (name, data) in enumerate(results.items(), 1):
        plt.subplot(1, len(results), i)
        
        # Scatter plot of predictions
        plt.scatter(y_test, data['predictions'], alpha=0.5, label='Predictions')
        
        # Add perfect prediction line
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', label='Perfect Prediction')
        
        r2_mean = data['metrics']['r2']['mean']
        r2_ci_lower = data['metrics']['r2']['ci_lower']
        r2_ci_upper = data['metrics']['r2']['ci_upper']
        
        plt.title(f'{name}\nR² = {r2_mean:.3f} ({r2_ci_lower:.3f}, {r2_ci_upper:.3f})')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
    
    plt.tight_layout()
    file_name = "./plots/predicted_vs_actual_values_model_comparison.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    
    # 3. Enhanced residuals box plot
    residuals_data = []
    for name, data in results.items():
        residuals = y_test - data['predictions']
        residuals_df = pd.DataFrame({
            'Model': name,
            'Residuals': residuals
        })
        residuals_data.append(residuals_df)
    
    residuals_df = pd.concat(residuals_data, ignore_index=True)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=residuals_df, x='Model', y='Residuals')
    plt.title('Residuals Distribution Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_name = "./plots/residuals_distribution_model_comparison.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    
    # Print detailed summary statistics
    print("\nModel Performance Summary with Confidence Intervals:")
    print("-" * 70)
    for name, data in results.items():
        print(f"\n{name}:")
        for metric in ['mse', 'mae', 'r2']:
            mean_val = data['metrics'][metric]['mean']
            ci_lower = data['metrics'][metric]['ci_lower']
            ci_upper = data['metrics'][metric]['ci_upper']
            print(f"{metric.upper()}: {mean_val:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    
    return results