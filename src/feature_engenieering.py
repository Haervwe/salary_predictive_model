
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler ,RobustScaler
import category_encoders as ce
import joblib
import os

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - df: Input DataFrame.

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets.
    """
    # Drop irrelevant features
    df = df.drop("Gender", axis=1)
    df = df.drop("id", axis=1)
    
    # Separate features and target variable
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def normalize_train_data(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    scaler: MinMaxScaler | StandardScaler | RobustScaler, 
    prefix: str = ""
) -> tuple[pd.DataFrame, ce.TargetEncoder, MinMaxScaler | StandardScaler | RobustScaler]:
    """
    Preprocesses the training data by encoding categorical variables,
    normalizing numerical variables, and target encoding 'Job Title'.

    Parameters:
    - X_train: Features of the training set.
    - y_train: Target variable of the training set.
    - scaler: Scaler object (MinMaxScaler, RobustScaller or StandardScaler).
    - prefix: Prefix for saved files (default is "").

    Returns:
    - X_train_processed: Preprocessed training features.
    - te: Fitted target encoder.
    - scaler: Fitted scaler for numerical features.
    """
    X_train = X_train.copy()
    y_train = y_train.copy()

    # Encode 'Education Level'
    education_order = {
        "Bachelor's": 0,
        "Master's": 1,
        "PhD": 2
    }
    X_train['Education Level'] = X_train['Education Level'].map(education_order)
    if X_train['Education Level'].isnull().any():
        X_train['Education Level'] = X_train['Education Level'].fillna(X_train['Education Level'].mode()[0])

    # Handle missing 'Job Title' values
    if X_train['Job Title'].isnull().any():
        X_train['Job Title'] = X_train['Job Title'].fillna('Unknown')

    # Target encode 'Job Title' without grouping
    smoothing = 10
    te = ce.TargetEncoder(cols=['Job Title'], smoothing=smoothing)
    te.fit(X_train[['Job Title']], y_train)  # Pass DataFrame with 'Job Title' column

    # Transform the 'Job Title' column
    X_train['Job Title Encoded'] = te.transform(X_train[['Job Title']])['Job Title']
    

    # Normalize numerical variables
    numeric_features = ['Age', 'Years of Experience', 'Job Title Encoded', 'Education Level']
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])

    # Save the fitted scaler, target encoder, and job title mapping
    os.makedirs('./models', exist_ok=True)  # Ensure the directory exists

    # Save scaler
    scaler_filename = f'./models/{prefix}scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to {scaler_filename}")

    # Save target encoder
    te_filename = f'./models/{prefix}target_encoder.pkl'
    joblib.dump(te, te_filename)
    print(f"Target encoder saved to {te_filename}")

    # Create and save a dictionary mapping job titles to their encoded values
    job_title_mapping = dict(zip(X_train['Job Title'], X_train['Job Title Encoded']))  # Correct mapping
    mapping_filename = f'./models/{prefix}job_title_mapping.pkl'
    joblib.dump(job_title_mapping, mapping_filename)
    print(f"Job title mapping saved to {mapping_filename}")
    
    # ***Create the inverted mapping***
    inverted_job_title_mapping = {v: k for k, v in job_title_mapping.items()}
    inverted_mapping_filename = f'./models/{prefix}inverted_job_title_mapping.pkl'
    joblib.dump(inverted_job_title_mapping, inverted_mapping_filename) 
    print(f"Inverted job title mapping saved to {inverted_mapping_filename}")
    X_train.drop('Job Title', axis=1, inplace=True)
    return X_train, te, scaler

def normalize_test_data(
    X_test: pd.DataFrame, 
    te: ce.TargetEncoder, 
    scaler: MinMaxScaler | StandardScaler, 
) -> pd.DataFrame:
    """
    Preprocesses the test data using the encoders and scaler fitted on the training data.

    Parameters:
    - X_test: Features of the test set.
    - te: Fitted target encoder from training data.
    - scaler: Fitted scaler from training data.
    - prefix: Prefix for saved files (default is "").

    Returns:
    - X_test_processed: Preprocessed test features.
    """
    X_test = X_test.copy()

    # Encode 'Education Level'
    education_order = {
        "Bachelor's": 0,
        "Master's": 1,
        "PhD": 2
    }
    X_test['Education Level'] = X_test['Education Level'].map(education_order)
    if X_test['Education Level'].isnull().any():
        X_test['Education Level'] = X_test['Education Level'].fillna(X_test['Education Level'].mode()[0])

    # Handle missing 'Job Title' values
    if X_test['Job Title'].isnull().any():
        X_test['Job Title'] = X_test['Job Title'].fillna('Unknown')

    # Transform 'Job Title' using the fitted target encoder
    X_test['Job Title Encoded'] = te.transform(X_test[['Job Title']])['Job Title']
    X_test.drop('Job Title', axis=1, inplace=True)

    # Normalize numerical variables using scaler fitted on training data
    numeric_features = ['Age', 'Years of Experience', 'Job Title Encoded', 'Education Level']
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    return X_test


def save_datasets(X_train, X_test, y_train, y_test, normalized_X_train, normalized_X_train_nn, normalized_X_test, normalized_X_test_nn):
    """
    Saves the preprocessed datasets to PKL files.

    Parameters:
    - X_train: Preprocessed training features.
    - X_test: Preprocessed test features.
    - y_train: Training target variable.
    - y_test: Test target variable.

    Returns:
    - None
    """
    os.makedirs('./datasets', exist_ok=True)  # Ensure the directory exists

    # Save preprocessed datasets
    X_train.to_pickle("./data/X_train.pkl")
    print(f"X_train saved to './datasets' directory.")
    y_train.to_pickle("./data/y_train.pkl")
    print(f"y_train saved to './datasets' directory.")
    X_test.to_pickle("./data/X_test.pkl")
    print(f"X_test saved to './datasets' directory.")  
    y_test.to_pickle("./data/y_test.pkl")
    print(f"y_test saved to './datasets' directory.")
    normalized_X_train.to_pickle("./data/normalized_X_train.pkl")
    print(f"normalized_X_train saved to './datasets' directory.")
    normalized_X_train_nn.to_pickle("./data/normalized_X_train_nn.pkl")
    print(f"normalized_X_train_nn saved to './datasets' directory.")
    normalized_X_test.to_pickle("./data/normalized_X_test.pkl")
    print(f"normalized_X_test saved to './datasets' directory.")
    normalized_X_test_nn.to_pickle("./data/normalized_X_test_nn.pkl")
    print(f"normalized_X_test_nn saved to './datasets' directory.")
    print(f"All Preprocessed datasets saved to './datasets' directory.")
    
    return None


def load_datasets():
    """
    Loads the preprocessed datasets from PKL files.

    Parameters:
    - None

    Returns:
    - X_train: Preprocessed training features.
    - X_test: Preprocessed test features.
    - y_train: Training target variable.
    - y_test: Test target variable.
    """
    X_train = pd.read_pickle("./data/X_train.pkl")
    y_train = pd.read_pickle("./data/y_train.pkl")
    X_test = pd.read_pickle("./data/X_test.pkl")
    y_test = pd.read_pickle("./data/y_test.pkl")
    normalized_X_train = pd.read_pickle("./data/normalized_X_train.pkl")
    normalized_X_train_nn = pd.read_pickle("./data/normalized_X_train_nn.pkl")
    normalized_X_test = pd.read_pickle("./data/normalized_X_test.pkl")
    normalized_X_test_nn = pd.read_pickle("./data/normalized_X_test_nn.pkl")
    return X_train, X_test, y_train, y_test, normalized_X_train, normalized_X_train_nn, normalized_X_test, normalized_X_test_nn