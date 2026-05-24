"""Helper functions for evaluating models."""

import functools

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, log_loss, f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import cross_validate

from src.plot.plot import make_and_save_plots


def append_array_to_scores(scores, metric_array, name):
    """
    Append an array of scores to a scores dictionary. Elements of the array are appended as key:value pairs.

    :param dict scores: scoring dictionary
    :param np.array metric_array: array of scores
    :param str name: name of metric
    :return: dictionary of scores
    :rtype: dict
    """
    for i, score in enumerate(metric_array):
        scores[f'{name}_{i}'] = score
    return scores


def compile_scores(y, y_pred, y_pred_proba, decision_boundary=0.5):
    """
    Compile a dictionary of evaluation metrics.

    :param pd.Series y: target series
    :param pd.Series y_pred: predicted target series
    :param pd.Series y_pred_proba: predicted probabilities series
    :param float decision_boundary: probability threshold for determining if a datapoint belongs to the positive or
        negative class; defaults to 0.5
    :return: evaluation metrics
    :rtype: dict
    """
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10,
                                             strategy='quantile')
    scores = {
        'datapoints': len(y),
        'decision_boundary': decision_boundary,
        'neg_brier_score': -brier_score_loss(y, y_pred_proba),
        'neg_log_loss': -log_loss(y, y_pred_proba),
        'f1': f1_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'accuracy': accuracy_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }
    scores = append_array_to_scores(scores, prob_true, 'prob_true_bin')
    scores = append_array_to_scores(scores, prob_pred, 'prob_pred_bin')
    return scores


def custom_scorer(pipeline, x, y, decision_boundary=0.5):
    """
    Score model using a variety of metrics.

    :param sklearn.pipeline.Pipeline pipeline: pipeline to evaluate
    :param pd.DataFrame x: features dataframe
    :param pd.Series y: target series
    :param float decision_boundary: probability threshold for determining if a datapoint belongs to the positive or
        negative class; defaults to 0.5
    :return: evaluation metrics
    :rtype: dict
    """
    y_pred_proba = pd.Series(pipeline.predict_proba(x)[:, 1], index=y.index)
    y_pred = pd.Series(np.where(y_pred_proba > decision_boundary, 1, 0), index=y.index)
    scores = compile_scores(y, y_pred, y_pred_proba, decision_boundary=decision_boundary)
    return scores


def evaluate_model(pipeline, x, y, estimator_name, results_path, decision_boundary=0.5, use_cv_for_eval=False, cv=5):
    """
    Evaluate model using a variety of metrics.

    :param sklearn.pipeline.Pipeline pipeline: pipeline to evaluate
    :param pd.DataFrame x: features dataframe
    :param pd.Series y: target series
    :param str estimator_name: the name of the final estimator in the pipeline
    :param pathlib.Path results_path: path to the model results directory
    :param float decision_boundary: probability threshold for determining if a datapoint belongs to the positive or
        negative class; defaults to 0.5
    :param bool use_cv_for_eval: Boolean for whether to use cross-validation
        for evaluation; defaults to False
    :param Union[int, None] cv: cross-validation scheme; only relevant if use_cv_for_eval=True; defaults to 5
    :return: None
    :rtype: None
    """
    if use_cv_for_eval:
        custom_scorer_partial = functools.partial(custom_scorer, decision_boundary=decision_boundary)
        scores = cross_validate(pipeline, x, y, cv=cv, scoring=custom_scorer_partial)
        scores = pd.DataFrame(scores).T
        num_folds = scores.shape[1]
        scores.columns = [f'fold_{i + 1}' for i in range(num_folds)]
        scores.index.name = 'metric'
        scores['mean'] = scores.mean(axis=1)
        scores['std'] = scores.std(axis=1)
        scores = scores[['mean']] + [col for col in scores.columns if col != 'mean']
    else:
        scores = custom_scorer(pipeline, x, y, decision_boundary=decision_boundary)
        for key, value in scores.items():
            scores[key] = [value]
        scores = pd.DataFrame(scores).T
        scores.index.name = 'metric'
        scores.columns = ['value']
        scores.index = ['test_' + index for index in scores.index]
        scores.index.name = 'metric'
    scores.to_csv(results_path / f'{estimator_name}_scores.csv')
    make_and_save_plots(scores, estimator_name, results_path, decision_boundary=decision_boundary)
