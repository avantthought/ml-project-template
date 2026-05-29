"""Helper functions for hyperparameter optimization using the optuna library."""

from functools import partial

import optuna
import pandas as pd
from sklearn.model_selection import cross_validate


def train_pipeline_with_optuna(pipeline, x, y, estimator_name, cv_strategy, cv_scoring_dict, param_func,
                               iterations, save_scores=False, save_path=None):
    """
    Trains and optimizes hyperparameters of a pipeline using the optuna. Returns the fitted pipeline set with the
        best parameters determined from the optimization.
    Each trial of the optimization uses mean cross validation scores of the first metric in cv_scoring_dict. A csv
        of all the trials with scores can optionally be saved. The trials are sorted from best to worst.

    :param sklearn.pipeline.Pipeline pipeline: pipeline to optimize
    :param pd.DataFrame x: features dataframe
    :param pd.Series y: target series
    :param str estimator_name: the name of the final estimator in the pipeline
    :param Union[int, callable] cv_strategy: cross validation strategy
    :param Dict[str, str] cv_scoring_dict: dictionary of scoring metrics to use during cross-validation
    :param function param_func: parameter space (wrapped in a function) to search for optimal hyperparameters
        using optuna; the function needs "trial" as the only parameter with the (dictionary) parameter space returned
    :param int iterations: number of trials to run the hyperparameter optimization
    :param bool save_scores: whether to save the scores csv or not; default is False
    :param Union[pathlib.Path, None] save_path: path (from project root) where to save scores csv
        if truthy; default is None
    :return: fitted pipeline set with the best hyperparameters from the optimization
    :rtype: sklearn.pipeline.Pipeline
    """
    cv_scores_df = pd.DataFrame()

    def _make_cv_score_row_df(scores_array, name):
        cv_score_row_df = pd.DataFrame(scores_array).T
        cv_score_row_df.columns = 'fold_' + cv_score_row_df.columns.astype(str) + f'_{name}'
        cv_score_row_df[f'mean_{name}'] = cv_score_row_df.mean(axis=1)
        cv_score_row_df[f'std_{name}'] = cv_score_row_df.drop(columns=[f'mean_{name}']).std(axis=1)
        return cv_score_row_df

    def _objective_partial(trial, pipeline, x, y, cv_strategy, cv_scoring_dict, param_func):
        params = param_func(trial)
        estimator_param_names = pipeline.get_params().keys()
        new_mapping = {g: params[f] for f in params.keys() for g in estimator_param_names if g.endswith(f)}
        pipeline.set_params(**new_mapping)
        scores_dict = cross_validate(pipeline, x, y, cv=cv_strategy, scoring=cv_scoring_dict, n_jobs=-1)
        scores_df_list = []
        for score_key in list(scores_dict.keys())[2:]:
            name = score_key.split('_', 1)[1]
            score_df = _make_cv_score_row_df(scores_dict[score_key], name)
            scores_df_list.append(score_df)
        trial_cv_scores_row_df = pd.concat(scores_df_list, axis=1)
        temp_params_df = pd.DataFrame(params, index=[0])
        trial_cv_scores_row_df = pd.concat([temp_params_df, trial_cv_scores_row_df], axis=1)
        nonlocal cv_scores_df
        trial_cv_scores_row_df.index = [len(cv_scores_df)]
        mean_optimization_score = trial_cv_scores_row_df[
            f'mean_{list(cv_scoring_dict.keys())[0]}'
        ][len(cv_scores_df)]
        cv_scores_df = pd.concat([cv_scores_df, trial_cv_scores_row_df], axis=0)
        return mean_optimization_score

    study = optuna.create_study(direction='maximize')
    objective = partial(_objective_partial, pipeline=pipeline, x=x, y=y,
                        cv_strategy=cv_strategy, cv_scoring_dict=cv_scoring_dict, param_func=param_func)
    study.optimize(objective, n_trials=iterations)
    params = study.best_params
    estimator_param_names = pipeline.get_params().keys()
    best_params = {g: params[f] for f in params.keys() for g in estimator_param_names if g.endswith(f)}

    cv_scores_df = cv_scores_df.sort_values(by=[f'mean_{list(cv_scoring_dict.keys())[0]}'], ascending=False)
    cv_scores_df = cv_scores_df.reset_index(drop=True)
    cv_scores_df = cv_scores_df.reset_index()
    cv_scores_df = cv_scores_df.rename(columns={'index': 'ranking'})
    pipeline.set_params(**best_params)
    pipeline.fit(x, y)
    if save_scores:
        if save_path is not None:
            cv_scores_df.to_csv(f'{save_path}/{estimator_name}_optuna_cv_scores.csv', index=False)
    return pipeline
