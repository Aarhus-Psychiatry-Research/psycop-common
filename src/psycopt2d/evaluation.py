"""Functions for evaluating a model's predictions."""
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from wandb.sdk.wandb_run import Run as wandb_run  # pylint: disable=no-name-in-module

from psycopt2d.tables import generate_feature_importances_table
from psycopt2d.tables.performance_by_threshold import (
    generate_performance_by_positive_rate_table,
)
from psycopt2d.utils import (
    AUC_LOGGING_FILE_PATH,
    PROJECT_ROOT,
    positive_rate_to_pred_probs,
    prediction_df_with_metadata_to_disk,
)
from psycopt2d.visualization import (
    plot_auc_by_time_from_first_visit,
    plot_feature_importances,
    plot_metric_by_time_until_diagnosis,
    plot_performance_by_calendar_time,
)
from psycopt2d.visualization.sens_over_time import plot_sensitivity_by_time_to_outcome
from psycopt2d.visualization.utils import log_image_to_wandb


def log_feature_importances(
    cfg: DictConfig,
    pipe: Pipeline,
    feature_names: Iterable[str],
    run: wandb_run,
    save_path: Optional[Path] = None,
) -> dict[str, Path]:
    """Log feature importances to wandb."""
    # Handle EBM differently as it autogenerates interaction terms
    if cfg.model.model_name == "ebm":
        feature_names = pipe["model"].feature_names

    feature_importance_plot_path = plot_feature_importances(
        column_names=feature_names,
        feature_importances=pipe["model"].feature_importances_,
        top_n_feature_importances=cfg.evaluation.top_n_feature_importances,
        save_path=save_path,
    )

    # Log as table too for readability
    feature_importances_table = generate_feature_importances_table(
        feature_names=feature_names,
        feature_importances=pipe["model"].feature_importances_,
    )

    run.log({"feature_importance_table": feature_importances_table})

    return {"feature_importance": feature_importance_plot_path}


def log_auc_to_file(cfg: DictConfig, run: wandb_run, auc: Union[float, int]):
    """Log AUC to file."""
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


def evaluate_model(
    cfg,
    pipe: Pipeline,
    eval_df: pd.DataFrame,
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
        eval_df (pd.DataFrame): Evalaution split
        y_col_name (str): Label column name
        train_col_names (Iterable[str]): Column names for all predictors
        y_hat_prob_col_name (str): Column name containing pred_proba output
        run (wandb_run): WandB run to log to.
    """
    SAVE_DIR = PROJECT_ROOT / ".tmp"  # pylint: disable=invalid-name
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir()
    # Initialise relevant variables for the upcoming evaluation
    y = eval_df[y_col_name]  # pylint: disable=invalid-name
    y_hat_probs = eval_df[y_hat_prob_col_name]
    auc = round(roc_auc_score(y, y_hat_probs), 3)
    outcome_timestamps = eval_df[cfg.data.outcome_timestamp_col_name]
    pred_timestamps = eval_df[cfg.data.pred_timestamp_col_name]
    y_hat_int = np.round(y_hat_probs, 0)

    first_visit_timestamp = eval_df.groupby(cfg.data.id_col_name)[
        cfg.data.pred_timestamp_col_name
    ].transform("min")

    pred_proba_thresholds = positive_rate_to_pred_probs(
        pred_probs=y_hat_probs,
        positive_rate_thresholds=cfg.evaluation.positive_rate_thresholds,
    )

    print(f"AUC: {auc}")

    log_auc_to_file(cfg, run=run, auc=auc)

    # Tables
    # Performance by threshold
    performance_by_threshold_df = generate_performance_by_positive_rate_table(
        labels=y,
        pred_probs=y_hat_probs,
        positive_rate_thresholds=cfg.evaluation.positive_rate_thresholds,
        pred_proba_thresholds=pred_proba_thresholds,
        ids=eval_df[cfg.data.id_col_name],
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
        feature_names = train_col_names

        feature_importances_plot_dict = log_feature_importances(
            cfg=cfg,
            pipe=pipe,
            feature_names=train_col_names,
            run=run,
            save_path=SAVE_DIR / "feature_importances.png",
        )

        plots.update(feature_importances_plot_dict)

        # Log as table too for readability
        feature_importances_table = generate_feature_importances_table(
            feature_names=feature_names,
            feature_importances=pipe["model"].feature_importances_,
        )
        run.log({"feature_importance_table": feature_importances_table})

    # Add plots
    plots.update(
        {
            "sensitivity_by_time_by_threshold": plot_sensitivity_by_time_to_outcome(
                labels=y,
                y_hat_probs=y_hat_probs,
                pred_proba_thresholds=pred_proba_thresholds,
                outcome_timestamps=outcome_timestamps,
                prediction_timestamps=pred_timestamps,
                save_path=SAVE_DIR / "sensitivity_by_time_by_threshold.png",
            ),
            "auc_by_calendar_time": plot_performance_by_calendar_time(
                labels=y,
                y_hat=y_hat_probs,
                timestamps=pred_timestamps,
                bin_period="Y",
                metric_fn=roc_auc_score,
                y_title="AUC",
                save_path=SAVE_DIR / "auc_by_calendar_time.png",
            ),
            "auc_by_time_from_first_visit": plot_auc_by_time_from_first_visit(
                labels=y,
                y_hat_probs=y_hat_probs,
                first_visit_timestamps=first_visit_timestamp,
                prediction_timestamps=pred_timestamps,
                save_path=SAVE_DIR / "auc_by_time_from_first_visit.png",
            ),
            "f1_by_time_until_diagnosis": plot_metric_by_time_until_diagnosis(
                labels=y,
                y_hat=y_hat_int,
                diagnosis_timestamps=outcome_timestamps,
                prediction_timestamps=pred_timestamps,
                metric_fn=f1_score,
                y_title="F1",
                save_path=SAVE_DIR / "f1_by_time_until_diagnosis.png",
            ),
        },
    )

    # Save results to disk
    prediction_df_with_metadata_to_disk(df=eval_df, cfg=cfg)

    # Log all the figures to wandb
    for chart_name, chart_path in plots.items():
        log_image_to_wandb(chart_path=chart_path, chart_name=chart_name, run=run)
