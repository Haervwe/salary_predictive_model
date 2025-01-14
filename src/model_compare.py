import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compare_models(models_data: Dict[str, Tuple[object, pd.DataFrame, pd.Series]], y_test: pd.Series) -> Dict:
    """
    Compare multiple models' performance metrics and visualize the comparisons.
    Handles different normalizations for each model.
    
    Parameters:
    -----------
    models_data : Dict[str, Tuple[object, pd.DataFrame, pd.Series]]
        Dictionary where:
        - key: model name (str)
        - value: tuple containing (model, normalized_X_test, y_test)
        Example: {
            'Random Forest': (rf_model, rf_normalized_X_test, y_test),
            'Neural Network': (nn_model, nn_normalized_X_test, y_test)
        }
    y_test : pd.Series
        True target values for comparison
        
    Returns:
    --------
    Dict
        Dictionary containing comparison metrics and plots
    """
    # Store predictions and metrics for all models
    results = {}
    
    for name, (model, X_test_normalized, _) in models_data.items():
        # Get predictions
        y_pred = model.predict(X_test_normalized)
        if isinstance(y_pred, np.ndarray) and len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
            
        # Calculate metrics
        results[name] = {
            'predictions': y_pred,
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    
    # Create comparison visualizations
    
    # 1. Metrics comparison bar plot
    metrics_df = pd.DataFrame({
        name: {
            'MSE': metrics['mse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2']
        }
        for name, metrics in results.items()
    }).T
    
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 2. Predictions vs Actual scatter plot
    plt.figure(figsize=(15, 5))
    for i, (name, metrics) in enumerate(results.items(), 1):
        plt.subplot(1, len(results), i)
        plt.scatter(y_test, metrics['predictions'], alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f'{name}\nR² = {metrics["r2"]:.3f}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
    plt.tight_layout()
    plt.show()
    
    # 3. Residuals box plot
    residuals_df = pd.DataFrame({
        name: y_test - metrics['predictions']
        for name, metrics in results.items()
    })
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=residuals_df)
    plt.title('Residuals Distribution Comparison')
    plt.xlabel('Model')
    plt.ylabel('Residuals')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nModel Performance Summary:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"MSE: {metrics['mse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R²:  {metrics['r2']:.2f}")
    
    return results