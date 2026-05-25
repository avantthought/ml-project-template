"""Master function to fetch and preprocess data for modeling."""
# This is both a module and a script: The build function can be called in src.modeling.train.py to further be used in
# a modeling pipeline, or build.py can simply be run separately to build and save both the raw and processed data.

from src.config.config import (PROCESSED_DATA_DIR, RAW_DATA_DIR, LOAD_CACHED_DATA,
                               PROCESSED_DATA_FILENAME, RAW_DATA_FILENAME)
from src.data.process.process import process_raw_data
from src.data.raw.raw import fetch_raw_data
from src.utils.utils import make_dir


def build(raw_path, raw_data_filename, process_path, processed_data_filename, load_cached=False):
    """
    Builds and saves the raw and processed data for modeling.

    :param pathlib.Path raw_path: path to save the raw data
    :param str raw_data_filename: name of raw data file (excluding file extension)
    :param pathlib.Path process_path: path to save the processed data
    :param str processed_data_filename: name of processed data file (excluding file extension)
    :param bool load_cached: if true, load cached raw data; default is False
    :return: dataframe of fetched and wrangled data (pre modeling pipeline)
    :rtype: pd.DataFrame
    """
    make_dir(raw_path)
    make_dir(process_path)
    df = fetch_raw_data(raw_path, name=raw_data_filename, load_cached=load_cached)
    df = process_raw_data(df, process_path, name=processed_data_filename)
    return df


if __name__ == '__main__':
    build(
        raw_path=RAW_DATA_DIR,
        raw_data_filename=RAW_DATA_FILENAME,
        process_path=PROCESSED_DATA_DIR,
        processed_data_filename=PROCESSED_DATA_FILENAME,
        load_cached=LOAD_CACHED_DATA
    )
