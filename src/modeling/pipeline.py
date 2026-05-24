"""Place modeling pipeline function(s) here."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from src.modeling.process import drop_fields, FeaturesToDict


def create_pipeline(model, fields_to_drop):
    """
    Builds a sklearn modeling pipeline.

    :param sklearn.base.BaseEstimator model: instantiated model to be placed at the end of the pipeline
    :param List[str] fields_to_drop: list of column names to drop from input data
    :return: modeling pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('numeric_tranformer', numeric_transformer, selector(dtype_exclude=['category', 'object'])),
        ('categorical_transformer', categorical_transformer, selector(dtype_include=['category', 'object']))
    ])
    pipeline = Pipeline(steps=[
        ('column_dropper', FunctionTransformer(drop_fields, validate=False, kw_args={'fields': fields_to_drop})),
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


# NOTE: the structure of this function is tightly coupled with the structure of the create_pipeline() function above.
def pipeline_preprocessor_model_splitter(x, pipeline):
    """"
    Breaks off the fitted model at the end of given fitted pipeline and returns it along with the preprocessor part of
        the given dataframe x. Function is (unfortunately but necessarily) tightly coupled with the structure of the
        given model pipeline.
    This function is necessary when using DictVectorizer with categorical features with feature importance tools such
        as SHAP (each individual dummy variable from one column is treated separately in SHAP).

    :param pandas.DataFrame x: feature dataframe
    :param sklearn.pipeline.Pipeline pipeline: fitted pipeline
    :return: x transformed with the preprocessing steps in pipeline, and the fitted model at the end of the pipeline
    :rtype: tuple[pandas.DataFrame, sklearn.calibration.CalibratedClassifierCV]
    """
    pipeline_copy = deepcopy(pipeline)
    fitted_model = pipeline_copy.steps.pop(len(pipeline_copy) - 1)[1]
    num_cols = pipeline_copy.named_steps['column_dropper'].transformer(x).select_dtypes(
        include=[np.number]).columns.tolist()
    cat_cols = pipeline_copy.named_steps['preprocessor'].named_transformers_.get(
        'categorical_transformer').named_steps['dict_vectorizer'].feature_names_
    x_preprocessed = pd.DataFrame(pipeline_copy.transform(x), columns=num_cols + cat_cols)
    return x_preprocessed, fitted_model
