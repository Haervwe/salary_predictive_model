import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input , Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

def train_model(normalized_X_train, y_train, prefix: str = ""):
    """
    Train a Random Forest Regression Model and save it to disk.

    Parameters:
    - normalized_X_train: Preprocessed training features.
    - y_train: Training target variable.
    - prefix: Prefix for saved files (default is "").

    Returns:
    - best_rf_model: Trained Random Forest model.
    """
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [4, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [0.5, 'sqrt']
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

    # Ensure the directory exists
    os.makedirs('./models', exist_ok=True)

    # Save the Random Forest model
    model_filename = f'./models/{prefix}random_forest_model.pkl'
    joblib.dump(best_rf_model, model_filename)
    print(f"Random Forest model saved to {model_filename}")
    
    # ========================================
    # Plotting the Grid Search Results
    # ========================================

    # Convert cv_results_ to a DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Save the cv_results_ to CSV for reference
    results_filename = f'./plots/{prefix}grid_search_results.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"Grid search results saved to {results_filename}")

    # Convert negative mean absolute error to positive values
    results_df['mean_absolute_error'] = -results_df['mean_test_score']

    # Example Plot 1: Heatmap of Mean Absolute Error vs. n_estimators and max_depth
    pivot_table = results_df.pivot_table(
        values='mean_absolute_error',
        index='param_n_estimators',
        columns='param_max_depth'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap='viridis')
    plt.title('Grid Search Mean Absolute Error')
    plt.ylabel('Number of Estimators (n_estimators)')
    plt.xlabel('Max Depth')
    plt.tight_layout()
    heatmap_filename = f'./plots/{prefix}grid_search_heatmap.png'
    plt.savefig(heatmap_filename)
    plt.show()
    print(f"Grid search heatmap saved to {heatmap_filename}")
    plt.close()

    # Example Plot 2: Line Plot of Mean Absolute Error vs. n_estimators for different max_depth
    # Filter the DataFrame for specific values to reduce complexity
    subset_df = results_df[
        (results_df['param_min_samples_split'] == 4) &
        (results_df['param_min_samples_leaf'] == 1) &
        (results_df['param_max_features'] == 0.5)
    ]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=subset_df,
        x='param_n_estimators',
        y='mean_absolute_error',
        hue='param_max_depth',
        marker='o'
    )
    plt.title('Mean Absolute Error vs. n_estimators for different max_depth')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Number of Estimators (n_estimators)')
    plt.legend(title='Max Depth')
    plt.tight_layout()
    lineplot_filename = f'./plots/{prefix}grid_search_lineplot.png'
    plt.savefig(lineplot_filename)
    plt.show()
    print(f"Grid search line plot saved to {lineplot_filename}")
    plt.close()

    # Return the best model
    return best_rf_model

def train_NN_model(normalized_X_train, y_train, prefix: str = ""):
    """
    Train a Neural Network model and save it to disk.

    Parameters:
    - normalized_X_train: Preprocessed training features.
    - y_train: Training target variable.
    - prefix: Prefix for saved files (default is "").

    Returns:
    - model_nn: Trained Neural Network model.
    """
    optimizer = Adam(learning_rate=0.001)
    # Define the model
    model_nn = Sequential()
    # Add an Input layer specifying the shape
    model_nn.add(Input(shape=(normalized_X_train.shape[1],)))
    model_nn.add(Dense(64, activation='relu'))
    model_nn.add(Dropout(0.2))
    model_nn.add(Dense(32, activation='relu'))
    model_nn.add(Dropout(0.2))
    model_nn.add(Dense(16, activation='relu'))
    model_nn.add(Dense(1))  # Output layer for regression


    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Compile the model
    model_nn.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    history = model_nn.fit(
        normalized_X_train, y_train,
        epochs=1000,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Plot training and validation loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Save the plot
    os.makedirs('./plots', exist_ok=True)  # Ensure the directory exists
    plot_filename = f'./plots/{prefix}nn_training_loss.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Training loss plot saved to {plot_filename}")

    # Save the Neural Network model
    os.makedirs('./models', exist_ok=True)  # Ensure the directory exists
    model_filename = f'./models/{prefix}neural_network_model.keras'
    model_nn.save(model_filename)
    print(f"Neural Network model saved to {model_filename}")

    return model_nn


def train_dummy_regressor(X_train, y_train,prefix=""):
    """
    Train a dummy regressor on the training data.
    Parameters:
        X_train (pandas.DataFrame): The feature matrix for training.
        y_train (numpy.ndarray or pandas.Series): The target variable for training.
    Returns:
        DummyRegressor: A trained dummy regressor model.
    """
    
    # Create a dummy regressor and fit it to the data
    dr = DummyRegressor(strategy='mean')
    dr.fit(X_train, y_train)
    # Save the dummy Regressor
    os.makedirs('./models', exist_ok=True)  # Ensure the directory exists
    model_filename = f'./models/{prefix}dummy_reggresor_model.pkl'
    joblib.dump(dr, model_filename)
    print(f"Dummy Regressor saved as: {model_filename}")
    return dr

