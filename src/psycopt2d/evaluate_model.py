from typing import Iterable, Optional

import wandb
from sklearn.metrics import roc_auc_score

from psycopt2d.tables.performance_by_threshold import (
    generate_performance_by_threshold_table,
)


def evaluate_model(
    X,
    y: Iterable[int],
    y_hat_prob: Iterable[float],
    run: Optional[wandb.run],
    cfg,
):
    if run:
        run.log({"roc_auc_unweighted": round(roc_auc_score(y, y_hat_prob), 3)})
        run.log(
            {
                "performance_by_threshold": generate_performance_by_threshold_table(
                    labels=y,
                    pred_probs=y_hat_prob,
                    threshold_percentiles=cfg.evaluation.tables.performance_by_threshold.threshold_percentiles,
                    ids=X[cfg.data.id_col_name],
                    pred_timestamps=X[cfg.data.pred_timestamp_col_name],
                    outcome_timestamps=X[cfg.data.outcome_timestamp_col_name],
                ),
            },
        )
    else:
        print(f"AUC is: {round(roc_auc_score(y, y_hat_prob), 3)}")
