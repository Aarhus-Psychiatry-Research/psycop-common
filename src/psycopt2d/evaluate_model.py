from typing import Optional

import altair as alt
import pandas as pd
from sklearn.metrics import roc_auc_score

import wandb
from psycopt2d.tables.performance_by_threshold import (
    generate_performance_by_threshold_table,
)
from psycopt2d.utils import get_thresholds_by_pred_proba_percentiles
from psycopt2d.visualization.altair_utils import log_altair_to_wandb
from psycopt2d.visualization.sens_over_time import plot_sensitivity_by_time_to_outcome


def evaluate_model(
    cfg,
    eval_dataset: pd.DataFrame,
    y_col_name: str,
    y_hat_prob_col_name: str,
    run: Optional[wandb.run],
):
    y = eval_dataset[y_col_name]
    y_hat_probs = eval_dataset[y_hat_prob_col_name]
    auc = round(roc_auc_score(y, y_hat_probs), 3)
    outcome_timestamps = eval_dataset[cfg.data.outcome_timestamp_col_name]
    pred_timestamps = eval_dataset[cfg.data.pred_timestamp_col_name]

    alt.data_transformers.disable_max_rows()

    print(f"AUC: {auc}")

    # Log to wandb
    # Numerical metrics
    run.log({"roc_auc_unweighted": auc})

    # Tables
    ## Performance by threshold

    performance_by_threshold_df = generate_performance_by_threshold_table(
        labels=y,
        pred_probs=y_hat_probs,
        threshold_percentiles=cfg.evaluation.tables.threshold_percentiles,
        ids=eval_dataset[cfg.data.id_col_name],
        pred_timestamps=pred_timestamps,
        outcome_timestamps=outcome_timestamps,
    )
    run.log(
        {"performance_by_threshold": performance_by_threshold_df},
    )

    # Figures
    plots = {}

    ## Sensitivity by time to outcome
    threshold_percentiles = cfg.evaluation.tables.threshold_percentiles
    thresholds = get_thresholds_by_pred_proba_percentiles(
        pred_probs=y_hat_probs,
        threshold_percentiles=threshold_percentiles,
    )

    for i, threshold in enumerate(thresholds):
        log_title = f"{threshold_percentiles[i]}_sensitivity_by_time_to_outcome"

        plots.update(
            {
                log_title: plot_sensitivity_by_time_to_outcome(
                    labels=y,
                    y_hat_probs=y_hat_probs,
                    threshold=threshold,
                    outcome_timestamps=outcome_timestamps,
                    prediction_timestamps=pred_timestamps,
                ),
            },
        )

    ## Log all the figures to wandb
    for chart_name, chart_obj in plots.items():
        log_altair_to_wandb(chart=chart_obj, chart_name=chart_name, run=run)
