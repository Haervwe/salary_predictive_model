import pandas as pd
import numpy as np
from sklearn.model_selection import KFold ,train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import category_encoders as ce


def split_data(df:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    # Separate features and target variable
    X = df.drop('Salary', axis=1)
    y = df['Salary']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def normalize_train_data(X_train: pd.DataFrame, y_train: pd.Series)-> tuple[pd.DataFrame,ce.TargetEncoder,MinMaxScaler]:
    """
    Preprocesses the training data by encoding categorical variables,
    normalizing numerical variables, and target encoding 'Job Title'.

    Parameters:
    - X_train: Features of the training set.
    - y_train: Target variable of the training set.

    Returns:
    - X_train_processed: Preprocessed training features.
    - te: Fitted target encoder.
    - scaler: Fitted scaler for numerical features.
    """
    X_train = X_train.copy()
    y_train = y_train.copy()

    # 1. Encode 'Gender'
    gender_mapping = {'Male': 0, 'Female': 1}
    X_train['Gender'] = X_train['Gender'].map(gender_mapping)
    if X_train['Gender'].isnull().any():
        X_train['Gender'] = X_train['Gender'].fillna(X_train['Gender'].mode()[0])
        
    #X_train = X_train.drop("Gender", axis=1)

    # 2. Encode 'Education Level'
    education_order = {
        "High School": 0,
        "Associate's": 1,
        "Bachelor's": 2,
        "Master's": 3,
        "PhD": 4
    }
    X_train['Education Level'] = X_train['Education Level'].map(education_order)
    if X_train['Education Level'].isnull().any():
        X_train['Education Level'] = X_train['Education Level'].fillna(X_train['Education Level'].mode()[0])

    # 3. Handle missing 'Job Title' values
    if X_train['Job Title'].isnull().any():
        X_train['Job Title'] = X_train['Job Title'].fillna('Unknown')

    # 4. Target encode 'Job Title' without grouping
    smoothing = 10  # Adjust smoothing parameter as needed
    te = ce.TargetEncoder(cols=['Job Title'], smoothing=smoothing)
    te.fit(X_train[['Job Title']], y_train)  # Pass DataFrame with 'Job Title' column

    # Transform the 'Job Title' column
    X_train['Job Title Encoded'] = te.transform(X_train[['Job Title']])['Job Title']
    X_train.drop('Job Title', axis=1, inplace=True)

    # 5. Normalize numerical variables
    scaler = StandardScaler()
    numeric_features = ['Age', 'Years of Experience', 'Job Title Encoded', 'Education Level']
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])

    return X_train, te, scaler

def normalize_test_data(X_test: pd.DataFrame, te: ce.TargetEncoder, scaler: MinMaxScaler)-> pd.DataFrame:
    """
    Preprocesses the test data using the encoders and scaler fitted on the training data.

    Parameters:
    - X_test: Features of the test set.
    - te: Fitted target encoder from training data.
    - scaler: Fitted scaler from training data.

    Returns:
    - X_test_processed: Preprocessed test features.
    """
    X_test = X_test.copy()

    # 1. Encode 'Gender'
    gender_mapping = {'Male': 0, 'Female': 1}
    X_test['Gender'] = X_test['Gender'].map(gender_mapping)
    if X_test['Gender'].isnull().any():
        X_test['Gender'] = X_test['Gender'].fillna(X_test['Gender'].mode()[0])
    #X_test = X_test.drop("Gender", axis=1)
    # 2. Encode 'Education Level'
    education_order = {
        "High School": 0,
        "Associate's": 1,
        "Bachelor's": 2,
        "Master's": 3,
        "PhD": 4
    }
    X_test['Education Level'] = X_test['Education Level'].map(education_order)
    if X_test['Education Level'].isnull().any():
        X_test['Education Level'] = X_test['Education Level'].fillna(X_test['Education Level'].mode()[0])

    # 3. Handle missing 'Job Title' values
    if X_test['Job Title'].isnull().any():
        X_test['Job Title'] = X_test['Job Title'].fillna('Unknown')

    # 4. Transform 'Job Title' using the fitted target encoder
    X_test['Job Title Encoded'] = te.transform(X_test[['Job Title']])['Job Title']
    X_test.drop('Job Title', axis=1, inplace=True)

    # 5. Normalize numerical variables using scaler fitted on training data
    numeric_features = ['Age', 'Years of Experience', 'Job Title Encoded', 'Education Level']
    X_test[numeric_features]  = scaler.transform(X_test[numeric_features])

    return X_test


