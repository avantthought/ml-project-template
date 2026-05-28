"""NOTE: hyperopt is no longer maintained, and template uses optuna.py instead.
Helper functions for hyperparameter optimization using the hyperopt library."""

# import time
#
# from hyperopt import fmin, tpe, STATUS_OK, Trials
# from hyperopt.early_stop import no_progress_loss
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import cross_validate
#
#
# class Objective:
#     """
#     Objective function for hyperparameter optimization.
#
#     :param sklearn.base.BaseEstimator estimator: the base estimator to optimize
#     :param numpy.ndarray X: input data
#     :param numpy.ndarray y: target data
#     :param Union[int, callable] cv: cross-validation strategy
#     :param Union[str, callable, dict] scoring: scoring metric(s) to use; if this is a dictionary the first item is the
#         score used in optimization
#     """
#
#     def __init__(self, estimator, X, y, cv=5, scoring='f1'):
#         """
#         Initialize an object.
#         """
#         self.estimator = estimator
#         self.X = X
#         self.y = y
#         self.cv = cv
#         self.scoring = scoring
#
#     def fix_param_keys(self, params):
#         """
#         Add the necessary prefixes to the keys of a hyperparameter dictionary. Estimator hyperparameters have prefixes
#             attached to them when embedded in a pipeline or a calibrated classifier.
#
#         :param dict params: hyperparameters for sefl.estimator
#         :return: hyperparameters with prefixes appended
#         :rtype: dict
#         """
#         estimator_param_names = self.estimator.get_params().keys()
#         new_mapping = {g: params[f] for f in params.keys() for g in estimator_param_names if g.endswith(f)}
#         return new_mapping
#
#     def __call__(self, params):
#         """
#         Evaluate the estimator with the given hyperparameters.
#
#         :param dict params: hyperparameters to evaluate
#         :return: score metadata
#         :rtype: dict
#         """
#         params = self.fix_param_keys(params)
#         estimator = self.estimator.set_params(**params)
#         tic = time.time()
#         scores = cross_validate(estimator, self.X, self.y, cv=self.cv, scoring=self.scoring)
#         toc = time.time()
#         delta_t = round((toc - tic) / 60, 4)
#         loss = 1 - np.mean(scores[list(scores.keys())[2]].means())
#         scores.update({
#             'loss': loss,
#             'status': STATUS_OK,
#             'params': params,
#             'runtime_minutes': delta_t
#         })
#         return scores
#
#
# class Hyperoptimizer:
#     """"
#     Wrapper for hyperparameter optimization for a given objective and cross-validation scheme.
#
#     :param Objective objective: base function to optimize
#     :param dict space: search space for hyperparameters; should be a dictionary where keys are hyperparameter
#         names and values are hyperopt distributions
#     :param int max_evals: maximum number of evaluations to perform
#     :param int early_stop_n: the number of iterations with no improvement after which to stop
#     :param dict _best_params: best hyperparameters found after optimization
#     :param hyperopt.Trials _trials: trials object storing details of results
#     """
#
#     def __init__(self, objective, space, max_evals=1000, early_stop_n=10):
#         """
#         Initialize a Hyperoptimizer.
#         """
#         self.objective = objective
#         self.space = space
#         self.max_evals = max_evals
#         self.early_stop_n = early_stop_n
#         self._best_params = None
#         self._trials = Trials()
#
#     @property
#     def trials(self):
#         """
#         Full metadata for hyperoptimization trials.
#
#         :rtype: list[dict]
#         """
#         return self._trials.trials
#
#     @property
#     def results(self):
#         """
#         A list of dictionaries returned by "objective" during the search.
#
#         :rtype: list[dict]
#         """
#         return self._trials.results
#
#     @property
#     def losses(self):
#         """
#         Trial losses.
#
#         :rtype: list[float]
#         """
#         return self._trials.losses()
#
#     @property
#     def statuses(self):
#         """
#         Trial statuses.
#
#         :rtype: list[str]
#         """
#         return self._trials.statuses()
#
#     @property
#     def best_loss(self):
#         """
#         The best loss found during hyperoptimiaztion.
#
#         :rtype: float
#         """
#         return min(self.losses)
#
#     @property
#     def best_params(self):
#         """
#         The best hyperparameters found during hyperoptimization.
#
#         :rtype: dict
#         """
#         for r in self.results:
#             if r['loss'] == self.best_loss:
#                 return r['params']
#
#     def search(self):
#         """
#         Find the best set of model hyperparameters using hyperopt.
#
#         :return: None
#         :rtype: None
#         """
#         self._best_params = fmin(
#             self.objective,
#             space=self.space,
#             algo=tpe.suggest,
#             max_evals=self.max_evals,
#             trials=self._trials,
#             early_stop_fn=no_progress_loss(self.early_stop_n)
#         )
#         return self
#
#
# def create_hyperopt_scores_df(hyperopt_trials_list, model_name):
#     """
#     Returns a dataframe of hyperopt scores. Each row is one trial and consists of the model hyperparameters
#         used; results of every cv score for every fold (in addition to the mean and std of the folds), and the
#         runtime of trial. The rows of the dataframe are sorted by best loss (i.e. best set of params).
#     Function is heavily tightly coupled with the structure of the Hyperoptimizer class and the structure of the output
#         of the trials property.
#
#     :param List[dict] hyperopt_trials_list: output of the trials property (after running search() on an instance
#         of Hyperoptimizer); this list of dictionaries contains all the metadata and results of running hyperopt
#     :param str model_name: name of the model
#     :return: dataframe of hyperopt cv scores and params where each row is for one trial
#     :rtype: pandas.DataFrame
#     """
#     hyperopt_scores_df = pd.DataFrame()
#     for trial in hyperopt_trials_list:
#         trial_number = trial['tid']
#         cv_score_keys = [f for f in trial['result'].keys() if f.startswith('test')]
#         params = trial['result']['params']
#         row = pd.DataFrame(params, index=[trial_number])
#         row.columns = np.transpose(list(row.columns.str.split('__')))[-1]
#         row.columns = model_name + '__' + row.columns
#         row['runtime_minutes'] = trial['result']['runtime_minutes']
#         row['loss'] = trial['result']['loss']
#         for cv_score in cv_score_keys:
#             cv_score_name = cv_score.removeprefix('test_')
#             cv_score_row = pd.DataFrame(trial['result'][cv_score]).T
#             cv_score_row.columns = cv_score.removeprefix('test_') + '_fold_' + cv_score_row.columns.astype(str)
#             cv_score_row[cv_score_name + '_mean'] = [cv_score_row.mean(axis=1)[0]]
#             cv_score_row[cv_score_name + '_std'] = [cv_score_row.std(axis=1)[0]]
#             cv_score_row.index = [trial_number]
#             column_reorder = cv_score_row.columns.tolist()
#             column_reorder = [column_reorder[-2]] + [column_reorder[-1]] + column_reorder[:-2]
#             cv_score_row = cv_score_row[column_reorder]
#             row = pd.concat([row, cv_score_row], axis=1)
#         hyperopt_scores_df = pd.concat([hyperopt_scores_df, row], axis=0)
#     hyperopt_scores_df = hyperopt_scores_df.sort_values(by='loss', ascending=True)
#     # TODO: consider add ranking column or rename the index as ranking.
#     hyperopt_scores_df = hyperopt_scores_df.reset_index(drop=True)
#     hyperopt_scores_df = hyperopt_scores_df.drop(columns=['loss'])
#     return hyperopt_scores_df
