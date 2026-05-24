"""Model training script."""

from auto_shap import produce_shap_values_and_summary_plots
import joblib
import numpy as np
import pandas as pd

from src.config.config import (RESULTS_DIR, DATA_LIMIT, OPTIMIZATION_ITERATIONS_LIMIT, SHAP_SAMPELE_SIZE, TARGET_NAME,
                               TRAIN_BASELINE, TRAIN_ONLY_ONE_BASELINE, SHAP_ON_BOTH_TEST_AND_TRAIN, TEST_SET_SIZE,
                               DECISION_THRESHOLD, CV_SPLITS, CV_SCORING_DICT, NON_MODELING_FIELDS_TO_DROP)
from src.data.build import build
from src.config.estimators import MODEL_TRAINING_LIST
from src.modeling.evaluate import evaluate_model
from src.modeling.hyperoptimize import Hyperoptimize, Objective, create_hyperopt_scores_df
from src.modeling.pipeline import create_pipeline, pipeline_preprocessor_model_splitter
from src.modeling.process import (create_target, create_x_y_dataframes, create_test_train_splits,
                                  determine_iterations, determine_shap_sampling)
from src.utils.utils import make_dir, create_datetime_id


def master_train(results_path, model_training_list, target_name, non_modeling_fields_to_drop,
                 cv_splits, cv_scoring_dict, test_set_size=0.25, decision_threshold=0.5, data_limit=None,
                 train_baseline=True, train_only_one_baseline=False, optimization_iterations_limit=None,
                 shap_on_both_test_and_train=False, shap_sample_size=None):
    """
    Master modeling function for all models in the given model_training_list.

    :param pathlib.Path results_path: path where to save modeling results
    :param List[scr.config.estimators.model_config] model_training_list: list of 'model_config' named tuples, where
        each tuple corresponds ot a single model that has the model name, model object, hyperparameter grid,
        and number of optimization iterations
    :param str target_name: name of the target variable/column
    :param List[str] non_modeling_fields_to_drop: list of fields/columns to drop before modeling
    :param Union[int, callable] cv_splits: cross-validation strategy
    :param Dict[str, str] cv_scoring_dict: dictionary of scoring metrics to use during cross-validation
    :param float test_set_size: proportion of the dataset to include in the test split; default is 0.25
    :param float decision_threshold: threshold for classifying positive class; default is 0.5
    :param Union[int, None] data_limit: limit on the number of rows to use from the dataset; default is None (no limit)
    :param bool train_baseline: whether to train baseline model; default is True
    :param bool train_only_one_baseline: whether to train only one non-baseline model; default is False
    :param Union[int, None] optimization_iterations_limit: global limit on the number of hyperparameter optimization
        iterations that will overwrite each model-defined specific in the model_training_list;
        default is None (the model-specific number of iterations is used)
    :param bool shap_on_both_test_and_train: whether to compute SHAP values on both train and
        test sets; default is False (only on the test set)
    :param Union[int, None] shap_sample_size: sample size for SHAP value computation; default is None (no limit)
    """
    training_path = results_path / create_datetime_id()
    make_dir(training_path)
    model_training_list_adjusted = model_training_list.copy()
    if train_only_one_baseline:
        model_training_list_adjusted = [model_training_list_adjusted[0], model_training_list_adjusted[-1]]
    if not train_baseline:
        model_training_list_adjusted = model_training_list_adjusted[:-1]

    df = build() #NOTE: build will still need raw and processed data paths passed
    # assume target column is in the dataframe
    pass


if __name__ == '__main__':
    master_train(
        results_path=RESULTS_DIR,
        model_training_list=MODEL_TRAINING_LIST,
        target_name=TARGET_NAME,
        non_modeling_fields_to_drop=NON_MODELING_FIELDS_TO_DROP,
        cv_splits=CV_SPLITS,
        cv_scoring_dict=CV_SCORING_DICT,
        test_set_size=TEST_SET_SIZE,
        decision_threshold=DECISION_THRESHOLD,
        data_limit=DATA_LIMIT,
        train_baseline=TRAIN_BASELINE,
        train_only_one_baseline=TRAIN_ONLY_ONE_BASELINE,
        optimization_iterations_limit=OPTIMIZATION_ITERATIONS_LIMIT,
        shap_on_both_test_and_train=SHAP_ON_BOTH_TEST_AND_TRAIN,
        shap_sample_size=SHAP_SAMPELE_SIZE,
    )
