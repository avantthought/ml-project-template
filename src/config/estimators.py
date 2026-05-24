"""For storing modeling hyperparameter grids and models. In MODEL_TRAINING_LIST, place the baseline model as the last
item, and if there is preferred/best model, place it as the first item."""

from collections import namedtuple

from hyperopt import hp
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

BASELINE_CLASSIFIER_SPACE = {
    'C': hp.loguniform('C', -4, 2),
    'penalty': hp.choice('penalty', ['l1', 'l2']),
}

LIGHTGBM_CLASSIFIER_SPACE = {
    'n_estimators': hp.uniformint('n_estimators', [30, 300]),
    'learning_rate': hp.loguniform('learning_rate', -4, 0),
    'num_leaves': hp.uniformint('num_leaves', [15, 50]),
    'max_depth': hp.uniformint('max_depth', [3, 20]),
    'min_child_weight': hp.uniformint('min_child_weight', [2, 16]),
    'reg_alpha': hp.loguniform('reg_alpha', -3, 2),
    'reg_lambda': hp.loguniform('reg_lambda', -3, 2),
}

RANDOM_FOREST_CLASSIFIER_SPACE = {
    'n_estimators': hp.uniformint('n_estimators', [75, 400]),
    'max_depth': hp.uniformint('max_depth', [3, 20]),
    'min_samples_leaf': hp.uniformint('min_samples_leaf', [1, 10]),
    'max_features': hp.choice('max_features', ['sqrt', 'log2']),
}

XGBOOST_CLASSIFIER_SPACE = {
    'n_estimators': hp.uniformint('n_estimators', [30, 300]),
    'learning_rate': hp.loguniform('learning_rate', -4, 0),
    'max_depth': hp.uniformint('max_depth', [3, 20]),
    'min_child_weight': hp.uniformint('min_child_weight', [2, 16]),
    'reg_alpha': hp.loguniform('reg_alpha', -3, 2),
    'reg_lambda': hp.loguniform('reg_lambda', -3, 2),
}

model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_grid', 'iterations'})

MODEL_TRAINING_LIST = [
    model_named_tuple(
        model_name='xgboost',
        model=CalibratedClassifierCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        param_grid=XGBOOST_CLASSIFIER_SPACE,
        iterations=50,
    ),
    model_named_tuple(
        model_name='lightgbm',
        model=CalibratedClassifierCV(LGBMClassifier()),
        param_grid=LIGHTGBM_CLASSIFIER_SPACE,
        iterations=50,
    ),
    model_named_tuple(
        model_name='random_forest',
        model=CalibratedClassifierCV(RandomForestClassifier()),
        param_grid=RANDOM_FOREST_CLASSIFIER_SPACE,
        iterations=50,
    ),
    model_named_tuple(
        model_name='baseline_logistic_regression',
        model=CalibratedClassifierCV(LogisticRegression()),
        param_grid=BASELINE_CLASSIFIER_SPACE,
        iterations=50,
    ),
]
