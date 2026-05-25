"""Configuration settings (constants) for the project"""

from pathlib import Path

LOAD_CACHED_DATA = False
RAW_DATA_FILENAME = 'raw_data'
PROCESSED_DATA_FILENAME = 'processed_data'
PROJECT_WD = Path(__file__).parents[2]
RAW_DATA_DIR = PROJECT_WD / 'data' / 'raw_data'
PROCESSED_DATA_DIR = PROJECT_WD / 'data' / 'processed_data'
RESULTS_DIR = PROJECT_WD / 'data' / 'modeling_results'

# use None in the following three settings for no limits; keep them low to prototype faster
DATA_LIMIT = None
OPTIMIZATION_ITERATIONS_LIMIT = 5
SHAP_SAMPLE_SIZE = None

TARGET_NAME = 'target'
TRAIN_BASELINE = True
TRAIN_ONLY_ONE_NON_BASELINE = False
SHAP_ON_BOTH_TEST_AND_TRAIN = False
SAVE_PRODUCTION_PIPELINE = False
PRODUCTION_MODEL_NAME = None

TEST_SET_SIZE = 0.2
DECISION_BOUNDARY = 0.5
CV_SPLITS = 5
CV_SCORING_DICT = {  # scores recorded during hyperparameter optimization; only the first one is used to optimize
    'log_loss': 'neg_log_loss',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'auc': 'roc_auc',
    }

NON_MODELING_FIELDS_TO_DROP = []
