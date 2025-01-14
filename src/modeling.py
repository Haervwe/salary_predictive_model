import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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

    # Save the Random Forest model
    os.makedirs('./models', exist_ok=True)  # Ensure the directory exists
    model_filename = f'./models/{prefix}random_forest_model.pkl'
    joblib.dump(best_rf_model, model_filename)
    print(f"Random Forest model saved to {model_filename}")

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
    model_nn = Sequential()
    model_nn.add(Dense(64, input_dim=normalized_X_train.shape[1], activation='relu'))
    model_nn.add(Dense(32, activation='relu'))
    model_nn.add(Dense(1))
    model_nn.add(Dense(128, input_dim=normalized_X_train.shape[1], activation='relu'))
    model_nn.add(Dropout(0.2))
    model_nn.add(Dense(64, activation='relu'))
    model_nn.add(Dropout(0.2))
    model_nn.add(Dense(32, activation='relu'))
    model_nn.add(Dense(1))

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
    plt.close()
    print(f"Training loss plot saved to {plot_filename}")

    # Save the Neural Network model
    os.makedirs('./models', exist_ok=True)  # Ensure the directory exists
    model_filename = f'./models/{prefix}neural_network_model'
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
    print(f"Random Forest model saved to {model_filename}")
    return dr

