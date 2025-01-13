import pandas as pd
import numpy as np
from typing import List
from src import llm_dataset_filler




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
    
    print("\nCleansed Dataset:")
    print(cleansed_dataset.head())
    print(cleansed_dataset.tail())
    print(cleansed_dataset.info())
    #print("\nMissing Values in Full Dataset:")
    print(cleansed_dataset.isnull().sum())
    return cleansed_dataset




