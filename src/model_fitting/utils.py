"""
utility functions
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


def get_clf(clf, calibrate=False, calibrate_cv=5, **kwargs):
    clf_dict = {"nb": MultinomialNB,
                "rf": RandomForestClassifier,
                "en": ElasticNet,
                'ab': AdaBoostClassifier
                }
    if clf == "xg":
        from xgboost import XGBClassifier
        clf_dict["xg"] = XGBClassifier
    if clf in clf_dict:
        clf = clf_dict[clf]
    else:
        raise ValueError(f"classifier {clf} is not a valid classifier")
    if calibrate:
        clf = CalibratedClassifierCV(clf, cv=calibrate_cv)
    return clf


def resample(df, label_column, method="over", sampling_strategy=1, **kwargs):
    """
    method ('over'|'under'):
    """
    if method == "under":
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
        df, y = rus.fit_resample(df, df[label_column])
    elif method == "over":
        ros = RandomOverSampler(sampling_strategy=sampling_strategy)
        df, y = ros.fit_resample(df, df[label_column])
    elif method == "smote":
        ros = SMOTE(sampling_strategy=sampling_strategy)
        df, y = ros.fit_resample(df, df[label_column])
    elif method is None:
        pass
    else:
        raise ValueError("Invalid method {method}")
    return df


def add_def_args(func, def_args):
    """
    func (fun): function which to add defaults arguments to
    def_args (dict): default argument given as a dictionary

    Examples
    >>> def addn(x,n):
    ...    return x + n
    >>> add3 = add_def_args(addn, {'n':3})
    >>> add3(2)
    5
    """
    def func_wrapper(*args, **kwargs):
        value = func(*args, **def_args, **kwargs)
        return value
    return func_wrapper