import pandas as pd
import numpy as np
from typing import Dict
from src import llm_dataset_filler
import pickle


# Detect outliers using IQR method
def detect_outliers_iqr(data:pd.DataFrame, multiplier:float=1.5)->Dict:
    numerical_cols = ['Age', 'Years of Experience', 'Salary']
    outliers = {}
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outlier_indices = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
        outliers[col] = outlier_indices
        print(f"Found {len(outlier_indices)} outliers in '{col}' using IQR method.")
    return outliers


#dataframe preprocessing
async def preprocess(full_dataset: pd.DataFrame, base_url:str=None, model_name:str=None, api_key:str="") -> pd.DataFrame:

    # Print initial dataset information
    print("\nFull Merged Dataset:")
    print(full_dataset.head())
    print(full_dataset.tail())
    print(full_dataset.info())

    # Check for missing values
    print("\nMissing Values in Full Dataset:")
    print(full_dataset.isnull().sum())

    # Display rows with missing values
    missing_rows = full_dataset[full_dataset.isnull().any(axis=1)]
    print("\nRows with Missing Values:")
    print(missing_rows)

    # Use an LLM to infer missing values from the 'Description' column
    if base_url is not None and model_name is not None:
        full_dataset = await llm_dataset_filler.infer_missing_values_in_dataframe(full_dataset,base_url=base_url, model_name=model_name, api_key=api_key)
    else:
        full_dataset = await llm_dataset_filler.infer_missing_values_in_dataframe(full_dataset)
    
    print("\nMissing Values after LLM inference:")
    print(full_dataset.isnull().sum())

    # Display rows that still have missing values and cannot be inferred
    missing_rows = full_dataset[full_dataset.isnull().any(axis=1)]
    print("\nRows with Missing Values after LLM inference:")
    print(missing_rows)

    # Replace "NotFound" with NaN
    full_dataset.replace("NotFound", np.nan, inplace=True)

    # Define critical columns for cleansing
    critical_columns = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']

    # Drop rows with any null values in critical columns
    cleansed_dataset = full_dataset.dropna(subset=critical_columns).reset_index(drop=True)

    # Remove the 'Description' column as it won't be used for training
    cleansed_dataset.drop('Description', axis=1, inplace=True)

    # Detect duplicate rows
    duplicate_rows = cleansed_dataset[cleansed_dataset.duplicated()]
    num_duplicates = duplicate_rows.shape[0]
    print(f"\nNumber of duplicate rows found: {num_duplicates}")
    if num_duplicates > 0:
        print("\nDuplicate rows:")
        print(duplicate_rows)

    # Remove duplicate rows
    cleansed_dataset = cleansed_dataset.drop_duplicates().reset_index(drop=True)
    print(f"Dataset shape after removing duplicates: {cleansed_dataset.shape}")

    # Detect outliers using IQR method
    outliers_iqr = detect_outliers_iqr(cleansed_dataset)
    all_outlier_indices = set()
    for indices in outliers_iqr.values():
        all_outlier_indices.update(indices)
    print(f"\nTotal number of outlier rows to remove: {len(all_outlier_indices)}")

    # Remove outlier rows
    cleansed_dataset = cleansed_dataset.drop(index=all_outlier_indices).reset_index(drop=True)
    print(f"Dataset shape after removing outliers: {cleansed_dataset.shape}")

    # Final dataset information
    print("\nCleansed Dataset after duplicate and outlier removal:")
    print(cleansed_dataset.head())
    print(cleansed_dataset.tail())
    print(cleansed_dataset.info())
    print("\nMissing Values in Cleansed Dataset:")
    print(cleansed_dataset.isnull().sum())

    # Save the cleansed dataset
    cleansed_dataset.to_pickle("./data/cleansed_dataset.pkl")
    return cleansed_dataset




