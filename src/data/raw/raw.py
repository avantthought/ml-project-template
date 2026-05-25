"""Helper functions for fetching raw data."""
# Insert functions to fetch raw data (both for modeling and production).
# In the template, the breast cancer data is used; change function(s) to fetch the desired data
# If many helper raw data functions are needed, consider splitting them into multiple modules in src.data.raw folder.

import joblib
from sklearn.datasets import load_breast_cancer  # delete this import when switching to desired data

from src.utils.utils import camel_to_snake


def fetch_raw_data(path, name='raw_data', load_cached=False):
    """
    Fetches and saves raw data.

    :param pathlib.Path path: path (from project root) to dump raw data (or to load cached data)
    :param str name: name of raw data file to be saved or loaded (excluding file extension)
    :param bool load_cached: if true, load cached data via path parameter; default is False
    :return: pandas dataframe of raw data
    :rtype: pd.DataFrame
    """
    if load_cached:
        df = joblib.load(path / f'{name}.pkl')
    else:
        df = load_breast_cancer(as_frame=True)['data']
        df['target'] = load_breast_cancer(as_frame=True)['target']
        df.columns = [camel_to_snake(col) for col in df.columns]
        df.columns = [col.replace(' ', '_') for col in df.columns]
        joblib.dump(df, path / f'{name}.pkl')
    return df
