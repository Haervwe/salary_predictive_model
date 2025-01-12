import pandas as pd
from typing import List



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



