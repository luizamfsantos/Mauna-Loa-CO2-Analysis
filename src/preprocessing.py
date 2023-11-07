import pandas as pd
from datetime import datetime, timedelta
import re

def clean_data(data):
    '''
    This function should handle the pre-processing steps, such as dropping data points recorded as -99.99 and splitting the data into training and test datasets.
    '''
    df = data.copy()
    df = change_column_names(df)
    df, _ = convert_data_types(df)
    df['exact_date'] = get_exact_date(df['exact_date'])
    df = add_time_index(df)
    return df

def read_data_from_csv(url):
    """
    Reads data from a CSV file from the given URL without specifying column names.
    """
    try:
        df = pd.read_csv(url, header=None, delimiter=r"\s+")
        return df
    except Exception as e:
        print(f"An error occurred while reading the data: {e}")
        return None

def change_column_names(df):
    """
    Cleans the data by keeping only required columns, renaming them, and converting data types.
    """
    try:
        df = df[[2, 4]].rename(columns={2: 'exact_date', 4: 'CO2_concentration'})
        return df
    except Exception as e:
        print(f"An error occurred while changing the column names of: {e}")
        return None

def remove_non_numeric_chars(s):
    """
    Removes non-numeric characters except '-' from a string.
    """
    return re.sub(r'[^-0-9.]', '', s)

def convert_data_types(df):
    """
    Converts the data types of the columns to the appropriate types.
    """
    error_rows = []
    try:
        df['exact_date'] = df['exact_date'].apply(remove_non_numeric_chars).apply(pd.to_numeric, errors='coerce')
        df['CO2_concentration'] = df['CO2_concentration'].apply(remove_non_numeric_chars).apply(pd.to_numeric, errors='coerce')
        return df, error_rows
    except Exception as e:
        print(f"An error occurred while converting the data types: {e}")
        error_rows.append(df.index)
        return None, error_rows


def get_exact_date(col, start_date=datetime(1900, 1, 1)):
    '''
    This function gets the exact date when the CO2 concentration was recorded.
    '''
    if not isinstance(start_date, datetime):
        raise ValueError("The start_date must be an instance of the datetime class.")
    if not isinstance(col, pd.Series):
        raise ValueError("The input column must be a pandas Series.")

    target_date = start_date + pd.to_timedelta(col, unit='D')
    return target_date


def add_time_index(df, date_col='exact_date'):
    '''
    Create column t where 0 is Jan 01 1958
    Every month is 1
    Every half month is 0.5 (for example Feb 17 1958 is 1.5)

    Parameters
    ----------
    df: pandas DataFrame
        contains column exact_date with datetime objects
        or a different column name if specified in date_col
    date_col: str
        name of column containing datetime objects

    Returns
    -------
    df: pandas DataFrame
        contains column t with time index
    '''
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    assert isinstance(date_col, str), "date_col must be a string"
    assert date_col in df.columns, f"df must contain column {date_col}"
    assert all(isinstance(val, datetime) for val in df[date_col]), f"column {date_col} must contain datetime objects"
    base_date = datetime(1958, 1, 1)
    df['t'] = df[date_col].apply(lambda x: (x.year - base_date.year) * 12 + (x.month - base_date.month) + (x.day - base_date.day) / 30)
    # round to 0, 0.5 or 1
    df['t'] = df['t'].apply(lambda x: round(x * 2) / 2)
    return df

def test_train_split(df):
    '''
    Split the data 80-20, since this is a time series, we should not use random split.
    '''
    return df.iloc[:int(len(df) * 0.8)], df.iloc[int(len(df) * 0.8):]


if __name__ == '__main__':
    url = "data/processed/CO2_no_heading.csv"
    df = read_data_from_csv(url)
    df = clean_data(df)
    train, test = test_train_split(df)
    print(df.head())
