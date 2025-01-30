import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    demographic_parity_ratio,
)
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

if __name__ == "__main__":
    pass