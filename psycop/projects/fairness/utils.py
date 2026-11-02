import numpy as np

from fairlearn.metrics import (
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_negative_rate,
    true_positive_rate,
)
from sklearn.metrics import precision_score, roc_auc_score


def na_auroc(y_true, y_pred, sample_weight=None):
    if np.unique(y_true).size < 2:
            return np.nan
    
    return roc_auc_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )

def na_precision(y_true, y_pred, sample_weight=None):
    return precision_score(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            zero_division=np.nan
        )
    
    
def na_positive_metric(metric):
    def wrapper(y_true, y_pred, sample_weight=None):
        if np.sum(np.asarray(y_true) == 1) == 0:
            return np.nan

        return metric(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )

    return wrapper

def na_negative_metric(metric):
    def wrapper(y_true, y_pred, sample_weight=None):
        if np.sum(np.asarray(y_true) == 0) == 0:
            return np.nan

        return metric(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )

    return wrapper