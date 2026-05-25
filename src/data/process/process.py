"""Helper functions to process the raw data"""
# Insert functions to process the raw data, such as data wrangling related functions or feature engineering
# Don't include MODELING helper processing functions (place these in src.modeling.process.py instead).
# If many processing functions are needed, consider splitting them into multiple modules in the src.data.process folder.

import joblib


def process_raw_data(df, path, name='processed_data'):
    """
    Master data cleaning and wrangling function. Saves and returns processed data.

    :param pd.DataFrame df: data to be processed
    :param pathlib.Path path: path (from project root) to dump processed data
    :param str name: name of processed data file to be saved (excluding file extension)
    :return: dataframe of cleaned/wrangled data (pre modeling pipeline)
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()

    # Insert data processing here, including cleaning, feature engineering, etc.

    joblib.dump(df_copy, path / f'{name}.pkl')
    return df_copy
