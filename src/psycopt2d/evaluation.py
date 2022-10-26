"""Functions for evaluating a model's predictions."""
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from sklearn.metrics import roc_auc_score
from wandb.sdk.wandb_run import Run as wandb_run  # pylint: disable=no-name-in-module
from wasabi import Printer

from psycopt2d.tables.performance_by_threshold import (
    generate_performance_by_positive_rate_table,
)
from psycopt2d.tables.tables import generate_feature_importances_table
from psycopt2d.utils.configs import FullConfig
from psycopt2d.utils.utils import PROJECT_ROOT, positive_rate_to_pred_probs
from psycopt2d.visualization import plot_feature_importances
from psycopt2d.visualization.utils import log_image_to_wandb


def log_feature_importances(
    cfg: DictConfig,
    feature_importance_dict: dict[str, float],
    run: wandb_run,
    save_path: Optional[Path] = None,
) -> dict[str, Path]:
    """Log feature importances to wandb."""
    feature_importance_plot_path = plot_feature_importances(
        feature_names=feature_importance_dict.keys(),
        feature_importances=feature_importance_dict.values(),
        top_n_feature_importances=cfg.eval.top_n_feature_importances,
        save_path=save_path,
    )

    # Log as table too for readability
    feature_importances_table = generate_feature_importances_table(
        feature_names=feature_importance_dict.keys(),
        feature_importances=feature_importance_dict.values(),
    )

    run.log({"feature_importance_table": feature_importances_table})

    return {"feature_importance": feature_importance_plot_path}


def evaluate_model(
    cfg: FullConfig,
    eval_df: pd.DataFrame,
    y_col_name: str,
    y_hat_prob_col_name: str,
    run: wandb_run,
    feature_importance_dict: Optional[dict[str, float]],
) -> None:
    """Runs the evaluation suite on the model and logs to WandB.

    Args:
        cfg (OmegaConf): The hydra config from the run
        pipe (Pipeline): Pipeline including the model
        eval_df (pd.DataFrame): Evalaution split
        y_col_name (str): Label column name
        train_col_names (Iterable[str]): Column names for all predictors
        y_hat_prob_col_name (str): Column name containing pred_proba output
        run (wandb_run): WandB run to log to.
        feature_importance_dict (Optional[dict[str, float]]): Dict of feature
            names and their importance. If None, will not log feature importance.
        selected_features (Optional[list[str]]): List of selected features after preprocessing.
            Used for plotting.
    """
    msg = Printer(timestamp=True)

    msg.info("Starting model evaluation")

    SAVE_DIR = PROJECT_ROOT / ".tmp"  # pylint: disable=invalid-name
    # When running tests in parallel with pytest-xdist,
    # this causes issues since multiple processes
    # override the same dir at once.
    # Can be solved by allowing config to override this
    # and using tmp_dir in pytest. Not worth refactoring
    # right now, though.

    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir()

    # Initialise relevant variables for the upcoming evaluation
    y = eval_df[y_col_name]  # pylint: disable=invalid-name
    y_hat_probs = eval_df[y_hat_prob_col_name]
    auc = round(roc_auc_score(y, y_hat_probs), 3)
    outcome_timestamps = eval_df[cfg.data.outcome_timestamp_col_name]
    pred_timestamps = eval_df[cfg.data.pred_timestamp_col_name]
    y_hat_int = np.round(y_hat_probs, 0)

    date_bins_ahead: Iterable[int] = cfg.eval.date_bins_ahead
    date_bins_behind: Iterable[int] = cfg.eval.date_bins_behind

    # Drop date_bins_direction if they are further away than min_lookdirection_days
    if cfg.data.min_lookbehind_days:
        date_bins_behind = [
            b for b in date_bins_behind if cfg.data.min_lookbehind_days > b
        ]

    if cfg.data.min_lookahead_days:
        date_bins_ahead = [
            b for b in date_bins_ahead if cfg.data.min_lookahead_days > abs(b)
        ]

    # Invert date_bins_behind to negative if it's not already
    if min(date_bins_behind) >= 0:
        date_bins_behind = [-d for d in date_bins_behind]

    # Sort date_bins_behind and date_bins_ahead to be monotonically increasing if they aren't already
    date_bins_behind = sorted(date_bins_behind)

    pred_proba_thresholds = positive_rate_to_pred_probs(
        pred_probs=y_hat_probs,
        positive_rate_thresholds=cfg.eval.positive_rate_thresholds,
    )

    msg.info(f"AUC: {auc}")
    run.log(
        {
            "roc_auc_unweighted": auc,
        },
    )

    # Tables
    # Performance by threshold
    performance_by_threshold_df = generate_performance_by_positive_rate_table(
        positive_rate_thresholds=cfg.evaluation.positive_rate_thresholds,
        pred_proba_thresholds=pred_proba_thresholds,
    )
    run.log(
        {"performance_by_threshold": performance_by_threshold_df},
    )

    # Figures
    plots = {}

    # Feature importance
    if feature_importance_dict is not None:
        feature_importances_plot_dict = log_feature_importances(
            cfg=cfg,
            feature_importance_dict=feature_importance_dict,
            run=run,
            save_path=SAVE_DIR / "feature_importances.png",
        )

        plots.update(feature_importances_plot_dict)

    # Log all the figures to wandb
    for chart_name, chart_path in plots.items():
        log_image_to_wandb(chart_path=chart_path, chart_name=chart_name, run=run)

    msg.info("Finished model evaluation, logging charts to WandB")
