"""Helper functions for plotting model evaluation results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plot.style import register_colormaps, set_plot_params

register_colormaps()


def test_plot():
    """
    Test the plot style with a dummy plot

    :return: None
    :rtype: None
    """
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    plt.plot(x, y)
    plt.savefig('test_plot.png')


def make_plot_data(scores, col_name):
    """
    Make data for plots

    :param pd.DataFrame scores: scores for model evaluation of shape (n_folds, n_scores)
    :param str col_name: name of column of scores dataframe from where to extract plot data
    :return: data to plot
    :rtype: tuple[pd.DataFrame, pd.DataFrame, np.array]
    """
    prob_true = scores.filter(regex='^test_prob_true_', axis=0)
    prob_pred = scores.filter(regex='^test_prob_pred_', axis=0)
    tn = scores.loc['test_tn', col_name]
    fp = scores.loc['test_fp', col_name]
    fn = scores.loc['test_fn', col_name]
    tp = scores.loc['test_tp', col_name]
    count = scores.loc['test_datapoints', col_name]
    conf_matrix_scores = np.array([[tp, fp],
                                   [fn, tn]])
    return prob_true, prob_pred, conf_matrix_scores, count


def plot_cv_calibration(prob_true, prob_pred, name, save_path):
    """
    Plot calibration curves and errors.

    :param pd.DataFrame prob_true: true probabilities of shape (n_samples, n_folds)
    :param pd.DataFrame prob_pred: predicted probabilities of shape (n_samples, n_folds)
    :param str name: name of model
    :param str save_path: path to save plots
    :return: None
    :rtype: None
    """
    fold_names = [col for col in prob_true.columns if col.startswith('fold')]
    num_of_folds = len(fold_names)
    fig, axs = plt.subplots(nrows=1, ncols=2,
                            sharey=True,
                            figsize=(9, 4))
    colors = plt.get_cmap('hellafresh')
    for i in [0, 1]:
        axs[i].plot(
            [0, 1],
            [0, 1],
            linestyle='--',
            color='black',
            alpha=0.5,
            label='Perfect Calibration'
        )
    for i, col in enumerate(fold_names):
        axs[0].plot(
            prob_pred[col],
            prob_true[col],
            color=colors(i / num_of_folds),
            label=f'{col}'
        )
    axs[1].errorbar(
        prob_pred['mean'],
        prob_true['mean'],
        xerr=prob_pred['std'],
        yerr=prob_true['std'],
        fmt='none',
        label='Model Calibration'
    )
    for i in [0, 1]:
        axs[i].set_xlim([0.1, 0.9])
        axs[i].set_ylim([0.1, 0.9])
        axs[i].set_title(f'Calibration Curve ({name})')
        axs[i].set_xlabel('Predicted Probability')
        axs[0].set_ylabel('True Probability')
        axs[i].legend(loc='upper left', framealpha=0.0)
    fig.tight_layout()
    fig.savefig(f'{save_path}/{name}_calibration_curve.png')
    plt.clf()


def plot_calibration(prob_true, prob_pred, name, save_path):
    """
    Plot calibration curve for holdout/test data

    :param pd.DataFrame prob_true: true probabilities of shape (n_samples, 1)
    :param pd.DataFrame prob_pred: predicted probabilities of shape (n_samples, 1)
    :param str name: name of model
    :param str save_path: path to save plots
    :return: None
    :rtype: None
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = plt.get_cmap('hellafresh')
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        color='black',
        alpha=0.5,
        label='Perfect Calibration'
    )
    ax.plot(prob_pred, prob_true, color=colors(0.5), label='Model Calibration')
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.1, 0.9])
    ax.set_title(f'Calibration Curve ({name})')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')
    ax.legend(loc='upper left', framealpha=0.0)
    fig.tight_layout()
    fig.savefig(f'{save_path}/{name}_calibration_curve.png')
    plt.clf()


def plot_confusion_matrix(scores, name, save_path, decision_boundary=0.5, count=None):
    """
    Plot confusion matrix.

    :param np.array scores: scores from model evaluation of shape (2, 2)
    :param str name: name of model
    :param str save_path: path to save plots
    :param float decision_boundary: probability threshold for determining if a datapoint belongs to the positive or
        negative class; defaults to 0.5
    :param Union[float, None] count: Number of datapoints; if not None will be used in title of Confusion Matrix;
        defaults to None
    :return: None
    :rtype: None
    """
    scores = (scores / scores.sum()).round(3)
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=(4, 4))
    ax.imshow(scores, cmap='hellafresh')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    for i in [0, 1]:
        for j in [0, 1]:
            ax.text(i, j, scores[i, j],
                    ha='center', va='center',
                    colors='xkcd:off white',
                    fontsize=18
                    )
    if count is None:
        title_append = ''
    else:
        title_append = f' and count {int(count)}'
    ax.set_title(f'Confusion Matrix (decision boundary {decision_boundary}{title_append})')
    fig.tight_layout()
    fig.savefig(f'{save_path}/{name}_confusion_matrix.png')
    plt.clf()


def make_and_save_plots(scores, name, save_path, decision_boundary=0.5):
    """
    Make and save plots.

    :param pd.DataFrame scores: scores from model evaluation of shape
        (n_number_of_metrics, n_folds + 2) or (n_number_of_metrics, 1)
    :param str name: name of model
    :param str save_path: path to save plots
    :param float decision_boundary: probability threshold for determining if a datapoint belongs to the positive or
        negative class; defaults to 0.5
    :return: None
    :rtype: None
    """
    set_plot_params(font_scale=0.8)
    if scores.shape[1] > 1:
        prob_true, prob_pred, conf_matrix_scores, count = make_plot_data(scores, 'mean')
        plot_cv_calibration(prob_true, prob_pred, name, save_path)
    else:
        prob_true, prob_pred, conf_matrix_scores, count = make_plot_data(scores, 'value')
        plot_calibration(prob_true, prob_pred, name, save_path)
    plot_confusion_matrix(conf_matrix_scores, name, save_path, decision_boundary=decision_boundary, count=count)
