import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, log_loss
)

from psycop.common.cross_experiments.cross_project_catalogue import (
    CROSS_EXPERIMENTS_BASE_PATH,
    ModelCatalogue,
)

def classification_metrics(df, y_col='y', prob_col='y_hat_prob', ppr=0.02):
    y = df[y_col]
    y_prob = df[prob_col]
    
    threshold = np.quantile(y_prob, 1 - ppr)        # derive threshold from PPR
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    metrics = {
        'Threshold':   threshold,                    # useful to log
        'PPR':         (tp + fp) / len(y),           # should be ≈ ppr
        'Accuracy':    accuracy_score(y, y_pred),
        'Sensitivity':  tp / (tp + fn),
        'Specificity':  tn / (tn + fp),
        'F1':          f1_score(y, y_pred),
        'ROC-AUC':     roc_auc_score(y, y_prob),
        'TP':          tp,
        'FP':          fp,
        'TN':          tn,
        'FN':          fn,
        'FPR':         fp / (fp + tn),
        'FNR':         fn / (fn + tp),
        'PPV':         tp / (tp + fp),
        'NPV':         tn / (tn + fn),
    }

    return pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])

if __name__ == "__main__":

    exp_names = {"ECT": "ECT_partial_binary_auroc_tuning_2026-05-29_16-59-19",
                    "Restraint": "Restraint_partial_binary_auroc_tuning_2026-05-21_15-45-50",
                    "SCZ_BP": "SCZ_BP_partial_binary_auroc_tuning_2026-05-28_14-26-22",
                    "CVD": "CVD_partial_binary_auroc_tuning_2026-06-20_18-33-30",
                    "FAI": "FAI_partial_binary_auroc_tuning_2026-06-20_11-18-54"}
    metric = "partial_binary_auroc"

    # exp_names = {"ECT": "ECT_binary_average_precision_tuning_2026-06-01_18-34-05",
    #                 "Restraint": "Restraint_binary_average_precision_tuning_2026-06-01_18-34-05",
    #                 "SCZ_BP": "SCZ_BP_binary_average_precision_tuning_2026-06-01_23-25-55",
    #                 "CVD": "CVD_binary_average_precision_tuning_2026-06-22_08-51-18",
    #                 "FAI": "FAI_binary_average_precision_tuning_2026-06-22_13-51-21"}
    # metric = "binary_average_precision"

    # exp_names = {"ECT": "ECT_binary_auroc_tuning_2026-06-02_10-57-33",
    #                 "Restraint": "Restraint_binary_auroc_tuning_2026-06-02_10-57-33",
    #                 "SCZ_BP": "SCZ_BP_binary_auroc_tuning_2026-06-02_16-44-26",
    #                 "CVD": "CVD_binary_auroc_tuning_2026-06-21_14-44-07",
    #                 "FAI": "FAI_binary_auroc_tuning_2026-06-19_11-30-25"}
    # metric = "binary_auroc"

    project_pprs = {"ECT": 0.02, # TODO fh: get these directly from the getters instead
                    "Restraint": 0.01,
                    "SCZ_BP": 0.04,
                    "CVD": 0.05,
                    "FAI": 0.05}

    catalogue = ModelCatalogue(
        projects=["FAI"]
    )  # ["CVD", "ECT", "Restraint", "FAI", "SCZ_BP", "T2D"])
    orig_eval_dfs = catalogue.get_eval_dfs()

    for project, orig_eval_df in orig_eval_dfs.items():

        exp_eval_df = pd.read_parquet(f"{CROSS_EXPERIMENTS_BASE_PATH}/{exp_names[project]}/eval_df.parquet")
        
        metrics_orig = classification_metrics(df = orig_eval_df, ppr=project_pprs[project])
        metrics_exp = classification_metrics(df = exp_eval_df, ppr=project_pprs[project])


        print("hey")