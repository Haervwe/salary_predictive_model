from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def evaluate_model( normalized_X_test:pd.DataFrame,y_test:pd.Series,model)->None:
 
    y_pred_best_rf = model.predict(normalized_X_test)
    mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
    mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
    r2_best_rf = r2_score(y_test, y_pred_best_rf)

    print("Random Forest Regressor Performance After Hyperparameter Tuning:")
    print(f"Mean Squared Error (MSE): {mse_best_rf:.2f}")
    print(f"Mean Absolute Error (MAE): {mae_best_rf:.2f}")
    print(f"R-squared Score (R²): {r2_best_rf:.2f}")
