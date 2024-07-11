import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

class TransformDataset:
    def __init__(self, dataset_name: str):
        """
        Initialize the TransformDataset class.

        Args:
            dataset_name (str): The name of the dataset to load from Hugging Face.
        """
        try:
            self.dataset = load_dataset(dataset_name)
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")

    def hug_to_pandas(self) -> pd.DataFrame:
        """
        Convert Hugging Face dataset to a pandas DataFrame.

        Returns:
            pd.DataFrame: The converted DataFrame.
        """
        try:
            return pd.DataFrame(self.dataset["train"])
        except Exception as e:
            raise ValueError(f"Error converting dataset to DataFrame: {e}")

    def filter_df(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Filter the DataFrame to include only specified columns.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            cols (list): The list of columns to include.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        return df[cols]

    def add_past_year_item(self, df: pd.DataFrame, main_col: str) -> pd.DataFrame:
        """
        Add a column with the past year's data for each 'cik' and 'company'.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            main_col (str): The column name containing the data to shift.

        Returns:
            pd.DataFrame: The DataFrame with the added past year's data column.
        """
        df = df.sort_values(by=['cik', 'company', 'date'])
        df['past_year_data'] = df.groupby(['cik', 'company'])[main_col].shift(1)
        df=df.dropna()
        return df