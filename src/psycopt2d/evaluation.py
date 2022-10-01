"""Functions for evaluating a model's prredictions."""
from collections.abc import Iterable

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from wandb.sdk.wandb_run import Run as wandb_run

from psycopt2d.tables import generate_feature_importances_table
from psycopt2d.tables.performance_by_threshold import (
    generate_performance_by_positive_rate_table,
)
from psycopt2d.utils import AUC_LOGGING_FILE_PATH, positive_rate_to_pred_probs
from psycopt2d.visualization import (
    plot_auc_by_time_from_first_visit,
    plot_feature_importances,
    plot_metric_by_time_until_diagnosis,
    plot_performance_by_calendar_time,
)
from psycopt2d.visualization.altair_utils import log_altair_to_wandb
from psycopt2d.visualization.sens_over_time import plot_sensitivity_by_time_to_outcome


def evaluate_model(
    cfg,
    pipe: Pipeline,
    eval_dataset: pd.DataFrame,
    y_col_name: str,
    train_col_names: Iterable[str],
    y_hat_prob_col_name: str,
    run: wandb_run,
):
    """Runs the evaluation suite on the model and logs to WandB.
    At present, this includes:
    1. AUC
    2. Table of performance by pred_proba threshold
    3. Feature importance
    4. Sensitivity by time to outcome
    5. AUC by calendar time
    6. AUC by time from first visit
    7. F1 by time until diagnosis

    Args:
        cfg (OmegaConf): The hydra config from the run
        pipe (Pipeline): Pipeline including the model
        eval_dataset (pd.DataFrame): Evalaution split
        y_col_name (str): Label column name
        train_col_names (Iterable[str]): Column names for all predictors
        y_hat_prob_col_name (str): Column name containing pred_proba output
        run (wandb_run): WandB run to log to.
    """
    y = eval_dataset[y_col_name]
    y_hat_probs = eval_dataset[y_hat_prob_col_name]
    auc = round(roc_auc_score(y, y_hat_probs), 3)
    outcome_timestamps = eval_dataset[cfg.data.outcome_timestamp_col_name]
    pred_timestamps = eval_dataset[cfg.data.pred_timestamp_col_name]
    y_hat_int = np.round(y_hat_probs, 0)
    first_visit_timestamp = eval_dataset.groupby(cfg.data.id_col_name)[
        cfg.data.pred_timestamp_col_name
    ].transform("min")

    pred_proba_thresholds = positive_rate_to_pred_probs(
        pred_probs=y_hat_probs,
        positive_rate_thresholds=cfg.evaluation.positive_rate_thresholds,
    )

    alt.data_transformers.disable_max_rows()

    print(f"AUC: {auc}")

    # Log to wandb
    # Numerical metrics
    run.log({"roc_auc_unweighted": auc})

    # log AUC and run ID to a file to find the best run later
    # Only create the file if it doesn't exists (will be auto-deleted/moved after
    # syncing). This is to avoid creating a new file every time the script is run
    # e.g. during a hyperparameter seacrch.
    if cfg.project.wandb_mode in {"offline", "dryrun"}:
        if not AUC_LOGGING_FILE_PATH.exists():
            AUC_LOGGING_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            AUC_LOGGING_FILE_PATH.touch()
            AUC_LOGGING_FILE_PATH.write_text("run_id,auc\n")
        with open(AUC_LOGGING_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"{run.id},{auc}\n")

    # Tables
    # Performance by threshold
    performance_by_threshold_df = generate_performance_by_positive_rate_table(
        labels=y,
        pred_probs=y_hat_probs,
        positive_rate_thresholds=cfg.evaluation.positive_rate_thresholds,
        pred_proba_thresholds=pred_proba_thresholds,
        ids=eval_dataset[cfg.data.id_col_name],
        pred_timestamps=pred_timestamps,
        outcome_timestamps=outcome_timestamps,
    )
    run.log(
        {"performance_by_threshold": performance_by_threshold_df},
    )

    # Figures
    plots = {}

    # Feature importance
    if hasattr(pipe["model"], "feature_importances_"):
        # Handle EBM differently as it autogenerates interaction terms
        if cfg.model.model_name == "ebm":
            feature_names = pipe["model"].feature_names
        else:
            feature_names = train_col_names

        feature_importances_plot = plot_feature_importances(
            column_names=feature_names,
            feature_importances=pipe["model"].feature_importances_,
            top_n_feature_importances=cfg.evaluation.top_n_feature_importances,
        )
        plots.update(
            {"feature_importance": feature_importances_plot},
        )
        # Log as table too for readability
        feature_importances_table = generate_feature_importances_table(
            feature_names=feature_names,
            feature_importances=pipe["model"].feature_importances_,
        )
        run.log({"feature_importance_table": feature_importances_table})

    # Sensitivity by time to outcome
    plots.update(
        {
            "sensitivity_by_time_by_threshold": plot_sensitivity_by_time_to_outcome(
                labels=y,
                y_hat_probs=y_hat_probs,
                pred_proba_thresholds=pred_proba_thresholds,
                outcome_timestamps=outcome_timestamps,
                prediction_timestamps=pred_timestamps,
            ),
            "auc_by_calendar_time": plot_performance_by_calendar_time(
                labels=y,
                y_hat=y_hat_probs,
                timestamps=pred_timestamps,
                bin_period="M",
                metric_fn=roc_auc_score,
                y_title="AUC",
            ),
            "auc_by_time_from_first_visit": plot_auc_by_time_from_first_visit(
                labels=y,
                y_hat_probs=y_hat_probs,
                first_visit_timestamps=first_visit_timestamp,
                prediction_timestamps=pred_timestamps,
            ),
            "f1_by_time_until_diagnosis": plot_metric_by_time_until_diagnosis(
                labels=y,
                y_hat=y_hat_int,
                diagnosis_timestamps=outcome_timestamps,
                prediction_timestamps=pred_timestamps,
                metric_fn=f1_score,
                y_title="F1",
            ),
        },
    )

    # Log all the figures to wandb
    for chart_name, chart_obj in plots.items():
        log_altair_to_wandb(chart=chart_obj, chart_name=chart_name, run=run)
