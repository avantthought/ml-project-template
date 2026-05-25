"""Helper functions for data processing in model training."""

import pandas as pd
from sklearn.model_selection import train_test_split


def create_target(df, target_name):
    """
    Creates the target column in the given dataframe.

    :param pandas.DataFrame df: given dataframe from which to create the target
    :param str target_name: name of the target variable/column
    :return: the dataframe with additional target column
    :rtype: pandas.DataFrame
    """
    df_copy = df.copy()
    # df_copy[target_name] =  # INSERT LOGIC FOR CREATING TARGET (the template example already has the target)
    return df_copy


def determine_iterations(iterations_limit, model_config_iterations):
    """
    Determine the number of iterations for which to run the model. iterations_limit will override the
        model_config_iterations iterations limit if truthy and smaller.

    :param Union[int, None] iterations_limit: value that overrides the set model named tuple
        iterations (if smaller and not None); use None to not override
    :param int model_config_iterations: model tuple set number of iterations for hyperparameter optimization
    :return: number of iterations for which to run the model
    :rtype: int
    """
    if iterations_limit:
        if iterations_limit < model_config_iterations:
            return iterations_limit
    return model_config_iterations


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


def get_x_y_random_sample(x, y, n, reset_index=False):
    """
    Get a random sample of feature dataframe x and its respective target series y.

    :param pandas.DataFrame x: pandas dataframe of features
    :param pandas.Series y: pandas series target
    :param int n: number of samples to randomly select
    :param bool reset_index: bool for whether to reset sample index; default is False
    :return: x_subsample and y_subsample
    :rtype: tuple[pandas.DataFrame, pandas.Series]
    """
    name = y.name
    df_subsample = pd.DataFrame.sample(pd.concat([x, y], axis=1), n=n, random_state=42)
    if reset_index:
        df_subsample = df_subsample.reset_index(drop=True)
    y_subsample = df_subsample[name]
    x_subsample = df_subsample.drop(columns=[name])
    return x_subsample, y_subsample


def determine_shap_sampling(x_train, x_test, y_train, y_test, n=None, reset_index=False):
    """
    Samples both the training and the test sets for shap. Does not sample (i.e. uses all the data) if n is greater
        than the number of data points or if None is used for n.

    :param pandas.DataFrame x_train: x_train dataframe
    :param pandas.DataFrame x_test: x_test dataframe
    :param pandas.Series y_train: y_train series
    :param pandas.Series y_test: y_test series
    :param Union[int, None] n: integer for sampling data for shap; use None to skip sampling and use
        all data; default is None
    :param bool reset_index: bool for whether to reset sample index; default is False
    :return: x_train, x_test, y_train, and y_test (or a subsample of them)
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]
    """
    if not n:
        return x_train, x_test, y_train, y_test
    if n < len(y_test):
        x_test_sub, y_test_sub = get_x_y_random_sample(x=x_test, y=y_test, n=n, reset_index=reset_index)
    else:
        x_test_sub, y_test_sub = x_test, y_test
    if n < len(y_train):
        x_train_sub, y_train_sub = get_x_y_random_sample(x=x_train, y=y_train, n=n, reset_index=reset_index)
    else:
        x_train_sub, y_train_sub = x_train, y_train
    return x_train_sub, x_test_sub, y_train_sub, y_test_sub
