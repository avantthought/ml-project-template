"""For storing modeling hyperparameter grids and models. In MODEL_TRAINING_LIST, place the baseline model as the last
item, and if there is preferred/best model, place it as the first item."""

from collections import namedtuple

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def create_xgboost_trial(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 30, 300),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 16),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1e2, log=True),
    }
    return params


def create_lightgbm_trial(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 30, 300),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 16),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1e2, log=True),
    }
    return params


def create_random_forest_trial(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'max_depth': trial.suggest_int('max_depth', 2, 16),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt'])
    }
    return params


def create_baseline_classifier_trial(trial):
    params = {
        'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.001, 0.999),
    }
    return params


model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_func', 'iterations'})

MODEL_TRAINING_LIST = [
    model_named_tuple(
        model_name='xgboost',
        model=CalibratedClassifierCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        param_func=create_xgboost_trial,
        iterations=50,
    ),
    model_named_tuple(
        model_name='lightgbm',
        model=CalibratedClassifierCV(LGBMClassifier()),
        param_func=create_lightgbm_trial,
        iterations=50,
    ),
    model_named_tuple(
        model_name='random_forest',
        model=CalibratedClassifierCV(RandomForestClassifier()),
        param_func=create_random_forest_trial,
        iterations=50,
    ),
    model_named_tuple(
        model_name='baseline_logistic_regression',
        model=CalibratedClassifierCV(LogisticRegression(solver='saga')),
        param_func=create_baseline_classifier_trial,
        iterations=50,
    ),
]
