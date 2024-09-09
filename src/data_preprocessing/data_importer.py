import pandas as pd

from src.utils.csv_operations import combine_csv_files


def import_data(file_paths, limit=None):
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    combined_data = combine_csv_files(file_paths, limit)

    df = combined_data.iloc[:, :6]  # Select only the first 6 columns
    df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']

    df['date'] = pd.to_datetime(df['date'], unit='ms')

    return df
