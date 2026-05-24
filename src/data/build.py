"""Master function to fetch and preprocess data for modeling."""
# This is both a module and a script: The build function can be called in src.modeling.train.py to further be used in
# a modeling pipeline, or build.py can simply be run separately to build and save both the raw and processed data.
from src.config.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.utils import make_dir


# Import functions from src.data.raw.raw.py and src.data.process.process.py as needed


def build(raw_path, process_path):  # add more arguments depending on project needs
    """
    Builds and saves the raw and processed data for modeling.

    :param pathlib.Path raw_path: path to save the raw data
    :param pathlib.Path process_path: path to save the processed data
    """
    make_dir(raw_path)
    make_dir(process_path)
    # fetch and save raw data with funcs from src.data.raw.raw.py (use raw_path argument)
    # process and save (processed) data with funcs from src.data.process.process.py (use process_path argument)
    # also return final processed data, so build() can be called in src.modeling.train.py or elsewhere


if __name__ == '__main__':
    build(
        raw_path=RAW_DATA_DIR,
        process_path=PROCESSED_DATA_DIR
    )
