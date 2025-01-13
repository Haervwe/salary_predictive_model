from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
import matplotlib.pyplot as plt


def train_model(normalized_X_train, y_train):
    """Train a Random Forest Regression Model"""


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
    return best_rf_model


def train_NN_model(normalized_X_train, y_train):
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
        restore_best_weights=True)
    
    # Compile the model
    model_nn.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    history = model_nn.fit(normalized_X_train, y_train, epochs=1000, batch_size=32, validation_split=0.1)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Save the plot
    plt.savefig('plots/nn_training_loss.png', bbox_inches='tight')
    plt.close()
    
    return model_nn

