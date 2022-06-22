import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def performance_by_threshold(real_values, pred_probs, threshold):
    preds = np.where(pred_probs > threshold, 1, 0)

    CM = confusion_matrix(real_values, preds)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    n = TN + FN + TP + FP

    Prevalence = round((TP + FP) / n, 2)

    PPV = round(TP / (TP + FP), 4)
    NPV = round(TN / (TN + FN), 4)

    Sensitivity = round(TP / (TP + FN), 4)
    Specificity = round(TN / (TN + FP), 4)

    FPR = round(FP / (TN + FP), 4)
    FNR = round(FN / (TP + FN), 4)

    Accuracy = round((TP + TN) / n, 4)

    # Must return lists as values, otherwise pd.Dataframe requires setting indeces
    metrics_matrix = pd.DataFrame(
        {
            "prevalence": [Prevalence],
            "PPV": [PPV],
            "NPV": [NPV],
            "sensitivity": [Sensitivity],
            "specificity": [Specificity],
            "FPR": [FPR],
            "FNR": [FNR],
            "accuracy": [Accuracy],
        },
    )

    return metrics_matrix


def get_time_from_pos_to_diag_df(
    eval_df: pd.DataFrame,
    id_col_name: str = "dw_ek_borger",
    pred_prob_col_name: str = "pred_prob",
    pred_timestamp_col_name: str = "timestamp",
    outcome_timestamp_col_name: str = "timestamp_t2d_diag",
    positive_threshold: float = 0.5,
) -> pd.DataFrame:
    df = eval_df

    timestamp_cols = [col for col in df.columns if "timestamp" in col]

    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col])

    # Get min date among true positives
    df["true_positive"] = (df[pred_prob_col_name] >= positive_threshold) & (
        df[outcome_timestamp_col_name].notnull()
    )
    df = df[df["true_positive"]]

    df["timestamp_first_pos_pred"] = df.groupby(id_col_name)[
        pred_timestamp_col_name
    ].transform("min")

    df = df.drop_duplicates(
        subset=[id_col_name, "timestamp_first_pos_pred", outcome_timestamp_col_name],
    )

    # Compare to timestamp_t2d_diag
    df["undiagnosed_days_saved"] = (
        df[outcome_timestamp_col_name] - df["timestamp_first_pos_pred"]
    )

    df = df[
        [
            id_col_name,
            "timestamp_first_pos_pred",
            outcome_timestamp_col_name,
            "undiagnosed_days_saved",
        ]
    ]

    undiagnosed_days_saved = (
        (df["undiagnosed_days_saved"] / np.timedelta64(1, "D")).astype(int).agg("sum")
    )
    days_saved_per_true_positive = round(
        (df["undiagnosed_days_saved"] / np.timedelta64(1, "D")).astype(int).agg("mean"),
        1,
    )

    return {
        "undiagnosed_days_saved": undiagnosed_days_saved,
        "days_saved_per_true_positive": days_saved_per_true_positive,
    }
