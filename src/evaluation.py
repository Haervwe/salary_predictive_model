from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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