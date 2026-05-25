"""Model training script."""

from auto_shap import produce_shap_values_and_summary_plots
import joblib
import numpy as np
import pandas as pd

from src.config.config import (PROCESSED_DATA_DIR, RAW_DATA_DIR, LOAD_CACHED_DATA,
                               PROCESSED_DATA_FILENAME, RAW_DATA_FILENAME, RESULTS_DIR, DATA_LIMIT,
                               OPTIMIZATION_ITERATIONS_LIMIT, SHAP_SAMPLE_SIZE, TARGET_NAME,
                               TRAIN_BASELINE, TRAIN_ONLY_ONE_NON_BASELINE, SHAP_ON_BOTH_TEST_AND_TRAIN, TEST_SET_SIZE,
                               DECISION_BOUNDARY, CV_SPLITS, CV_SCORING_DICT, NON_MODELING_FIELDS_TO_DROP,
                               SAVE_PRODUCTION_PIPELINE, PRODUCTION_MODEL_NAME)
from src.data.build import build
from src.config.estimators import MODEL_TRAINING_LIST
from src.modeling.evaluate import evaluate_model
from src.modeling.hyperoptimize import Hyperoptimize, Objective, create_hyperopt_scores_df
from src.modeling.pipeline import create_pipeline, pipeline_preprocessor_model_splitter
from src.modeling.process import (create_target, create_x_y_dataframes, create_test_train_splits,
                                  determine_iterations, determine_shap_sampling)
from src.utils.utils import make_dir, create_datetime_id


def train_and_evaluate(training_path, model, model_name, param_grid, iterations, x_train, x_test, y_train, y_test,
                       non_modeling_fields_to_drop, cv_splits, cv_scoring_dict, decision_boundary=0.5,
                       shap_sample_size=None, shap_on_both_test_and_train=False, also_save_model_in_root=False):
    """
    Model training, evaluation, and explanation function for a single model. The trained model and all diagnostics
        are saved in a single directory determined by the given training_path and the model_name. A copy of the
        pipeline can additionally be saved in the project root directory (for production).

    :param pathlib.Path training_path: path (from project root) to dump model training results
    :param sklearn.base.BaseEstimator model: instantiated model
    :param str model_name: string name of the model
    :param dict param_grid: parameter space to search for optimal hyperparameters for the model
    :param int iterations: max number of trials to run the optimization
    :param pd.DataFrame x_train: x_train
    :param pd.DataFrame x_test: x_test
    :param pd.Series y_train: y_train
    :param pd.Series y_test: y_test
    :param List[str] non_modeling_fields_to_drop: list of columns to drop for modeling in pipeline
    :param Union[int, callable] cv_splits: cross validation strategy
    :param Dict[str, str] cv_scoring_dict: dictionary of scoring metrics to use during cross-validation
    :param float decision_boundary: threshold for classifying positive class; default is 0.5
    :param Union[int, None] shap_sample_size: sample size for SHAP value computation; default is None (no limit)
    :param bool shap_on_both_test_and_train: whether to compute SHAP values on both train and
        test sets; default is False (i.e., only used on the test set)
    :param bool also_save_model_in_root: if True, will also save the trained pipeline in
        the repository root (to be used in production); default is False
    :return: None
    :rtype: None
    """
    pass


def master_train(raw_path, raw_data_filename, process_path, processed_data_filename, results_path, model_training_list,
                 target_name, non_modeling_fields_to_drop,
                 cv_splits, cv_scoring_dict, test_set_size=0.25, decision_boundary=0.5, data_limit=None,
                 train_baseline=True, train_only_one_baseline=False, optimization_iterations_limit=None,
                 shap_on_both_test_and_train=False, shap_sample_size=None, save_production_pipeline=False,
                 production_model_name=None, load_cached=False):
    """
    Master modeling function for all models in the given model_training_list.

    :param pathlib.Path raw_path: path (from project root) to save the raw data
    :param str raw_data_filename: name of raw data file (excluding file extension)
    :param pathlib.Path process_path: path (from project root) to save the processed data
    :param str processed_data_filename: name of processed data file (excluding file extension)
    :param pathlib.Path results_path: path (from project root) where to save modeling results
    :param List[scr.config.estimators.model_config] model_training_list: list of 'model_config' named tuples, where
        each tuple corresponds ot a single model that has the model name, model object, hyperparameter grid,
        and number of optimization iterations
    :param str target_name: name of the target variable/column
    :param List[str] non_modeling_fields_to_drop: list of fields/columns to drop before modeling
    :param Union[int, callable] cv_splits: cross-validation strategy
    :param Dict[str, str] cv_scoring_dict: dictionary of scoring metrics to use during cross-validation
    :param float test_set_size: proportion of the dataset to include in the test split; default is 0.25
    :param float decision_boundary: threshold for classifying positive class; default is 0.5
    :param Union[int, None] data_limit: limit on the number of rows to use from the dataset; default is None (no limit)
    :param bool train_baseline: whether to train baseline model; default is True
    :param bool train_only_one_baseline: whether to train only one non-baseline model; default is False
    :param Union[int, None] optimization_iterations_limit: global limit on the number of hyperparameter optimization
        iterations that will overwrite each model-defined specific in the model_training_list;
        default is None (the model-specific number of iterations is used)
    :param bool shap_on_both_test_and_train: whether to compute SHAP values on both train and
        test sets; default is False (only on the test set)
    :param Union[int, None] shap_sample_size: sample size for SHAP value computation; default is None (no limit)
    :param bool save_production_pipeline: whether to save production pipeline; default is False
    :param Union[str, None] production_model_name: name of production model to be potentially saved
        in root; default is None
    :param bool load_cached: if true, load cached raw data; default is False
    :return: None
    :rtype: None
    """
    training_path = results_path / create_datetime_id()
    make_dir(training_path)
    model_training_list_adjusted = model_training_list.copy()
    if train_only_one_baseline:
        model_training_list_adjusted = [model_training_list_adjusted[0], model_training_list_adjusted[-1]]
    if not train_baseline:
        model_training_list_adjusted = model_training_list_adjusted[:-1]

    df = build(raw_path=raw_path,
               raw_data_filename=raw_data_filename,
               process_path=process_path,
               processed_data_filename=processed_data_filename,
               load_cached=load_cached)
    df = create_target(df, target_name)
    x, y = create_x_y_dataframes(df=df, target_name=target_name, data_limit=data_limit)
    train_test_dict = create_test_train_splits(x=x, y=y, test_set_size=test_set_size)
    joblib.dump(train_test_dict, f'{training_path}/train_test_dict.pkl')

    for model_tuple in model_training_list_adjusted:
        iterations = determine_iterations(optimization_iterations_limit, model_tuple.iterations)
        also_save_model_in_root = False
        if (model_tuple.model_name == production_model_name) and save_production_pipeline:
            also_save_model_in_root = True
        train_and_evaluate(
            training_path=training_path,
            model=model_tuple.model,
            model_name=model_tuple.model_name,
            param_grid=model_tuple.param_grid,
            iterations=iterations,
            x_train=train_test_dict['x_train'],
            x_test=train_test_dict['x_test'],
            y_train=train_test_dict['y_train'],
            y_test=train_test_dict['y_test'],
            non_modeling_fields_to_drop=non_modeling_fields_to_drop,
            cv_splits=cv_splits,
            cv_scoring_dict=cv_scoring_dict,
            decision_boundary=decision_boundary,
            shap_sample_size=shap_sample_size,
            shap_on_both_test_and_train=shap_on_both_test_and_train,
            also_save_model_in_root=also_save_model_in_root
        )


if __name__ == '__main__':
    master_train(
        raw_path=RAW_DATA_DIR,
        raw_data_filename=RAW_DATA_FILENAME,
        process_path=PROCESSED_DATA_DIR,
        processed_data_filename=PROCESSED_DATA_FILENAME,
        results_path=RESULTS_DIR,
        model_training_list=MODEL_TRAINING_LIST,
        target_name=TARGET_NAME,
        non_modeling_fields_to_drop=NON_MODELING_FIELDS_TO_DROP,
        cv_splits=CV_SPLITS,
        cv_scoring_dict=CV_SCORING_DICT,
        test_set_size=TEST_SET_SIZE,
        decision_boundary=DECISION_BOUNDARY,
        data_limit=DATA_LIMIT,
        train_baseline=TRAIN_BASELINE,
        train_only_one_baseline=TRAIN_ONLY_ONE_NON_BASELINE,
        optimization_iterations_limit=OPTIMIZATION_ITERATIONS_LIMIT,
        shap_on_both_test_and_train=SHAP_ON_BOTH_TEST_AND_TRAIN,
        shap_sample_size=SHAP_SAMPLE_SIZE,
        save_production_pipeline=SAVE_PRODUCTION_PIPELINE,
        production_model_name=PRODUCTION_MODEL_NAME,
        load_cached=LOAD_CACHED_DATA
    )
