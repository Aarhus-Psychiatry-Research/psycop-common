from typing import Optional

import pandas as pd
from sklearn.metrics import roc_auc_score

import wandb
from psycopt2d.tables.performance_by_threshold import (
    generate_performance_by_threshold_table,
)


def evaluate_model(
    cfg,
    eval_dataset: pd.DataFrame,
    y_col_name: str,
    y_hat_prob_col_name: str,
    run: Optional[wandb.run],
):
    y = eval_dataset[y_col_name]
    y_hat_prob = eval_dataset[y_hat_prob_col_name]
    if run:
        run.log({"roc_auc_unweighted": round(roc_auc_score(y, y_hat_prob), 3)})
        run.log(
            {
                "performance_by_threshold": generate_performance_by_threshold_table(
                    labels=y,
                    pred_probs=y_hat_prob,
                    threshold_percentiles=cfg.evaluation.tables.performance_by_threshold.threshold_percentiles,
                    ids=eval_dataset[cfg.data.id_col_name],
                    pred_timestamps=eval_dataset[cfg.data.pred_timestamp_col_name],
                    outcome_timestamps=eval_dataset[
                        cfg.data.outcome_timestamp_col_name
                    ],
                ),
            },
        )
    else:
        print(f"AUC is: {round(roc_auc_score(y, y_hat_prob), 3)}")
