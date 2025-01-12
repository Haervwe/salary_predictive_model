import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
from data_loading import load_data
from preprocessing import preprocess

def split_data(df:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    # Separate features and target variable
    X = df.drop('Salary', axis=1)
    y = df['Salary']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def normalize_train_data(df:pd.DataFrame)->pd.DataFrame:
    """Transform the data in to usable features for the ML model, 
    we transform each field in to a representation between 0 to 1 for the values that have ordinality implied, 
    and for the "job title" that has no ordianlity we use target encoding with regularization to capture the relationship between salary and job title without implicitly assuminyng ordinality per se.
    the target of the model in this case "salary" is leaved as an integer value, 
    since it's a continous variable that can take any real number.
    
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The preprocessed DataFrame ready for modeling.
    """

     # Copy the DataFrame to avoid modifying the original data
    df = df.copy()

    # 1. Encode 'Gender' as 0 (Male) and 1 (Female)
    gender_mapping = {'Male': 0, 'Female': 1}
    df['Gender'] = df['Gender'].map(gender_mapping)
    

    # 2. Normalize 'Age' and 'Years of Experience' between 0 and 1
    scaler = MinMaxScaler()
    numeric_features = ['Age', 'Years of Experience']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # 3. Encode 'Education Level' using ordinal encoding
    # Assuming an order in education levels
    education_order = {"High School": 0, "Associate's": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    df['Education Level'] = df['Education Level'].map(education_order)
    
    # Handle missing values in 'Education Level'
    if df['Education Level'].isnull().any():
        df['Education Level'].fillna(df['Education Level'].mode()[0], inplace=True)
    
    # 4. Target encode 'Job Title' with regularization

    # Initialize target encoder with smoothing (regularization)
    # Using cross-validation to avoid data leakage

    # Parameters for target encoding
    smoothing = 0.3 
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Initialize an empty array to store encoded values
    job_title_encoded = pd.Series(index=X.index, dtype=float)

    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]

        # Fit encoder on training fold
        te = ce.TargetEncoder(cols=['Job Title'], smoothing=smoothing)
        te.fit(X_train_fold['Job Title'], y_train_fold)

        # Transform validation fold
        job_title_encoded.iloc[val_idx] = te.transform(X_val_fold['Job Title'])['Job Title']

    # Assign the encoded 'Job Title' to the DataFrame
    df['Job Title'] = job_title_encoded

    # Handle any missing values in 'Job Title'
    if df['Job Title'].isnull().any():
        df['Job Title'].fillna(df['Job Title'].mean(), inplace=True)

    # Scale 'Job Title' to [0, 1]
    df['Job Title'] = scaler.fit_transform(df[['Job Title']])

    # 5. (Optional) Scale 'Education Level' to [0, 1]
    df['Education Level'] = scaler.fit_transform(df[['Education Level']])

    # At this point, all features are numerical and normalized between 0 and 1

    return df

def normalize_test_data(X_test:pd.DataFrame):
    """
    Preprocesses the test data using the encoders and scalers fitted on the training data.

    Returns the processed X_test.
    """
    X_test = X_test.copy()

    # 1. Encode 'Gender' as 0 (Male) and 1 (Female)
    gender_mapping = {'Male': 0, 'Female': 1}
    X_test['Gender'] = X_test['Gender'].map(gender_mapping)
    # Handle missing values
    if X_test['Gender'].isnull().any():
        X_test['Gender'].fillna(X_test['Gender'].mode()[0], inplace=True)

    # 2. Encode 'Education Level' using ordinal encoding
    education_order = {"High School": 0, "Associate's": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    X_test['Education Level'] = X_test['Education Level'].map(education_order)
    if X_test['Education Level'].isnull().any():
        X_test['Education Level'].fillna(X_test['Education Level'].mode()[0], inplace=True)

    # 3. Apply target encoder to 'Job Title'
    X_test['Job Title'] = te.transform(X_test['Job Title'])['Job Title']
    # The encoder handles unseen categories appropriately

    # 4. Normalize numerical variables using scaler fitted on training data
    numeric_features = ['Age', 'Years of Experience', 'Job Title', 'Education Level']
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_test



#files path for the raw dataset:
data_files = ['./data/people.csv','./data/descriptions.csv','./data/salary.csv',]

#merge datasets in a cohesive Dataframe
full_dataset = load_data(data_files)

#preprocessing of the dataframe adds missing values with LLM inference over descriptions of each row, drops the incomplete rows and cleans up the data.
cleansed_dataset = preprocess(full_dataset)

#split the dataset into an 80 / 20 ratio for training and testing.
datasets_tuple = split_data(cleansed_dataset) 
X_train, X_test, y_train, y_test = datasets_tuple
#normalize and scale the data
 
normalized_train_dataset = normalize_train_data(X_train)
normalized_test_dataset = normalize_test_data(X_test)

# Separate features and target variable



print(normalized_test_dataset)
print(normalized_train_dataset)