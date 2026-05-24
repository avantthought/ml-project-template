"""Configuration settings (constants) for the project"""

from pathlib import Path

PROJECT_WD = Path(__file__).parents[2]
RAW_DATA_DIR = PROJECT_WD / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_WD / 'data' / 'processed'
RESULTS_DIR = PROJECT_WD / 'data' / 'results'

# use None in the following three settings for no limits; keep them low to prototype faster
DATA_LIMIT = None
OPTIMIZATION_ITERATIONS_LIMIT = None
SHAP_SAMPELE_SIZE = None

TARGET_NAME = 'target'
TRAIN_BASELINE = True
TRAIN_ONLY_ONE_BASELINE = True
SHAP_ON_BOTH_TEST_AND_TRAIN = False
TEST_SET_SIZE = 0.2
DECISION_THRESHOLD = 0.5
CV_SPLITS = 5
CV_SCORING_DICT = {  # scores recorded during hyperparameter optimization; only the first one is used to optimize
    'log_loss': 'neg_log_loss',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'auc': 'roc_auc',
    }

NON_MODELING_FIELDS_TO_DROP = []
