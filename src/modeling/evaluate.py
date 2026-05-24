"""Helper functions for evaluating models."""

import functools

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, log_loss, f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import cross_validate

