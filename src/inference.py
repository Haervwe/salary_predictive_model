import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import category_encoders as ce


def make_inference_nn(input_data: pd.DataFrame, prefix: str = "") -> np.ndarray:
    """
    Makes inference on a Keras Neural Network model using preprocessed input data.

    Args:
        input_data: DataFrame containing the input data for inference.
        prefix: Prefix used for saving/loading models and preprocessors.

    Returns:
        Numpy array of predicted salary values.
    """

    # Load the scaler
    scaler_filename = f'./models/{prefix}scaler.pkl'
    scaler = joblib.load(scaler_filename)

    # Load the target encoder
    te_filename = f'./models/{prefix}target_encoder.pkl'
    te = joblib.load(te_filename)

    print(input_data)
    # Preprocess the input data
    input_data_processed = preprocess_inference_data(input_data.copy(), te, scaler)
    print(input_data_processed)

    # Load the trained Neural Network model
    model_filename = f'./models/{prefix}neural_network_model.keras'

    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")

    model_nn = tf.keras.models.load_model(model_filename)

    # Make predictions
    predictions = model_nn.predict(input_data_processed)
    
    return predictions


def preprocess_inference_data(
    X: pd.DataFrame,
    te: ce.TargetEncoder,
    scaler: MinMaxScaler | StandardScaler,
) -> pd.DataFrame:
    """
    Preprocesses the inference data using the provided encoder and scaler.
    """
        # Drop irrelevant features
    X = X.drop("Gender", axis=1, errors='ignore') #ignore if not present
    X = X.drop("id", axis=1, errors='ignore')
    
    # Encode 'Education Level'
    education_order = {
        "Bachelor's": 0,
        "Master's": 1,
        "PhD": 2
    }
    X['Education Level'] = X['Education Level'].map(education_order)
    if X['Education Level'].isnull().any():
        X['Education Level'] = X['Education Level'].fillna(X['Education Level'].mode()[0])


    # Handle missing 'Job Title' values, fill with 'Unknown' if TargetEncoder can handle unseen values
    if X['Job Title'].isnull().any():
        X['Job Title'] = X['Job Title'].fillna('Unknown')


    # Transform 'Job Title' using the fitted target encoder
    X['Job Title Encoded'] = te.transform(X[['Job Title']])['Job Title']
        
    X.drop('Job Title', axis=1, inplace=True)


    # Normalize numerical variables
    numeric_features = ['Age', 'Years of Experience', 'Job Title Encoded', 'Education Level']
    X[numeric_features] = scaler.transform(X[numeric_features])


    return X






def get_unique_job_titles(prefix=""):
    """Loads the unique job titles from the training data.

    Args:
          prefix (str, optional): Prefix used when saving the job title mapping. Defaults to "".


    Returns:
        list: A list of unique job titles.  Returns an empty list if there was a problem loading the job titles.
    """
    try:
        # Load job title mapping
        mapping_filename = f'./models/{prefix}job_title_mapping.pkl'
        job_title_mapping = joblib.load(mapping_filename)
        # Extract unique job titles from the keys, as the mapping might not contain all originally possible titles
        unique_job_titles = list(job_title_mapping.keys())
        unique_job_titles.sort() #makes the dropdown more predictable
        return unique_job_titles
    except Exception as e:
        print(f"Error loading job titles: {e}")
        return [] # Return empty list to prevent further errors