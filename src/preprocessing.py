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

async def preprocess(full_dataset:pd.DataFrame)->pd.DataFrame:

    #prints to check dataset merge
    print("\nFull Merged Dataset:")
    print(full_dataset.head())
    print(full_dataset.tail())
    print(full_dataset.info())


    #Check missing values and print

    print("\nMissing Values in Full Dataset:")
    print(full_dataset.isnull().sum())

    # Display rows with missing values
    missing_rows = full_dataset[full_dataset.isnull().any(axis=1)]
    print("\nRows with Missing Values:")
    print(missing_rows)

    #use an LLM to infer missing values form description column.
    full_dataset = await llm_dataset_filler.infer_missing_values_in_dataframe(full_dataset)
    print("\nMissing Values in Full Dataset:")

    #print Rows that cant be infered and cleanse them
    print(full_dataset.isnull().sum())
    missing_rows = full_dataset[full_dataset.isnull().any(axis=1)]
    print("\nRows with Missing Values:")
    print(missing_rows)
    
    #repace "NotFound" with "np.nan"
    full_dataset.replace("NotFound", np.nan, inplace=True)
    
    #drop all rows that have any null values in the relvant fields
    
    critical_columns = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']
    
    cleansed_dataset = full_dataset.dropna(subset=critical_columns).reset_index(drop=True)
    
    #remove description column since its a high dimesional feature and will not be used to train the model
    cleansed_dataset.drop('Description', axis=1, inplace=True)

    outliers_iqr = detect_outliers_iqr(cleansed_dataset)
    all_outlier_indices = set()
    for indices in outliers_iqr.values():
        all_outlier_indices.update(indices)
    print(all_outlier_indices)
    # Remove outlier rows
    cleansed_dataset = cleansed_dataset.drop(index=all_outlier_indices).reset_index(drop=True)
    print(f"Dataset shape after removing outliers: {cleansed_dataset.shape}")
    print("\nCleansed Dataset after outlier removal:")
    print(cleansed_dataset.head())
    print(cleansed_dataset.tail())
    print(cleansed_dataset.info())
    print(cleansed_dataset.isnull().sum())
    cleansed_dataset.to_pickle("./data/cleansed_dataset.pkl")
    return cleansed_dataset




