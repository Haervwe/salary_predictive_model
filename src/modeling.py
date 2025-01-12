from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def train_model(normalized_X_train, y_train):
    """Train a Random Forest Regression Model"""


    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [1.0, 'sqrt']
    }

    # Initialize the Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42)

    # Initialize Grid Search
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2
    )

    # Fit Grid Search on the training data
    grid_search.fit(normalized_X_train, y_train)

    # Best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)

    # Best model
    best_rf_model = grid_search.best_estimator_
    return best_rf_model


