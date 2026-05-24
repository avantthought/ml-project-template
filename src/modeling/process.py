"""Helper functions for data processing in modeling."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


def drop_fieds(df, fields):
    """
    Drops the given fields from a dataframe. sklearn-pipeline compatible function.

    :param pandas.DataFrame df: input dataframe
    :param List[str] fields: list of column names to drop from given dataframe
    :return: dataframe with given fields dropped
    :rtype: pandas.DataFrame
    """
    return df.drop(columns=fields, errors='ignore')


def create_x_y_dataframes(df, target_name, data_limit=None):
    """
    Creates X (features) and y (target) dataframes/series from a given dataframe df. The given target_name is required
        to be in the dataframe. Optionally, a data_limit can be provided to limit the number of rows for modeling.

    :param pandas.DataFrame df: input dataframe
    :param str target_name: name of the target variable/column
    :param Union[int, None] data_limit: limit on the number of rows to use from the dataset; default is None (no limit)
    :return: tuple of features dataframe and target series
    :rtype: Tuple[pandas.DataFrame, pandas.Series]
    """
    df_copy = df.copy()
    if data_limit:
        df_copy = df_copy.head(data_limit)
    y = df_copy[target_name]
    x = df_copy.drop(columns=[target_name])
    return x, y


def create_test_train_splits(x, y, test_set_size=0.25, tail_is_test_set=False, reset_index=False):
    """
    Creates a train-test split for modeling. By default, the sets are random; however, the tail of the data can be the
        test set (which is appropriate for time series or time-based cross validation).

    :param pandas.DataFrame x: pandas dataframe of features
    :param pandas.Series y: pandas series target
    :param float test_set_size: proportion of the dataset to include in the test split; default is 0.25
    :param bool tail_is_test_set: if True the test set is the tail of data; default is False (random test set)
    :param bool reset_index: boolean for whether to reset the index on the train-test split; default is False
    :return: dictionary of train-test split
    :rtype: dict[str, Union[pandas.DataFrame, pandas.Series]]
    """
    if tail_is_test_set:
        test_set_count = int(len(x) * test_set_size)
        x_train, x_test, y_train, y_test = (x.head(n=-test_set_count), x.tail(n=test_set_count),
                                            y.head(n=-test_set_count), y.tail(n=test_set_count))
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_size, random_state=42)
    if reset_index:
        x_test = x_test.reset_index(drop=True)
        x_train = x_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
    return {'x_test': x_test, 'x_train': x_train, 'y_test': y_test, 'y_train': y_train}


class FeaturesToDict(BaseEstimator, TransformerMixin):
    """
    Converts a dataframe or numpy array into a dict oriented by records. This is useful when using sklearn's
        DictVectorizer in a sklearn pipeline.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.to_dict(orient='records')
        return X
