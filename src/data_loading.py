import pandas as pd
from typing import List
import asyncio
from llm_dataset_filler import infer_missing_values_in_dataframe


def load_data(data_paths:list[str])->pd.DataFrame:
    """
    Loads multiple CSV files and merges them into a single DataFrame on the 'id' column.

    Parameters:
    - file_paths: List of file paths to CSV files.

    Returns:
    - full_dataset: A merged DataFrame containing data from all input files.
    """
    full_dataset = None
    for data in data_paths:
        df = pd.read_csv(data)
        ## print(df.head())
        ## print(df.info())
        if full_dataset is None:
            full_dataset = df.copy()
        else:
             # Merge on the 'id' column
            full_dataset = full_dataset.merge(df, how='outer')

    return full_dataset



data_files = ['./data/people.csv','./data/descriptions.csv','./data/salary.csv',]

full_dataset = load_data(data_files)
print("\nFull Merged Dataset:")
print(full_dataset.head())
print(full_dataset.tail())
print(full_dataset.info())
print("\nMissing Values in Full Dataset:")
#print(full_dataset.isnull())
print(full_dataset.isnull().sum())
# Display rows with missing values
missing_rows = full_dataset[full_dataset.isnull().any(axis=1)]
print("\nRows with Missing Values:")
print(missing_rows)
df = asyncio.run(infer_missing_values_in_dataframe(full_dataset))
print("\nMissing Values in Full Dataset:")
#print(full_dataset.isnull())
print(full_dataset.isnull().sum())
print(df)
missing_rows = full_dataset[full_dataset.isnull().any(axis=1)]
print("\nRows with Missing Values:")
print(missing_rows)