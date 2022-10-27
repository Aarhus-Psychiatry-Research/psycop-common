"""_summary_"""
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
import wandb
from sklearn.metrics import recall_score
from wandb.sdk.wandb_run import Run as wandb_run  # pylint: disable=no-name-in-module

from psycopt2d.evaluation_dataclasses import (
    ArtifactContainer,
    EvalDataset,
    PipeMetadata,
)
from psycopt2d.tables.performance_by_threshold import (
    generate_performance_by_positive_rate_table,
)
from psycopt2d.tables.tables import generate_feature_importances_table
from psycopt2d.utils.configs import FullConfig
from psycopt2d.utils.utils import positive_rate_to_pred_probs
from psycopt2d.visualization.feature_importance import plot_feature_importances
from psycopt2d.visualization.performance_over_time import (
    plot_auc_by_time_from_first_visit,
    plot_metric_by_calendar_time,
    plot_metric_by_time_until_diagnosis,
)
from psycopt2d.visualization.sens_over_time import (
    plot_sensitivity_by_time_to_outcome_heatmap,
)
from psycopt2d.visualization.utils import log_image_to_wandb


def upload_artifacts(
    artifact_containers: Iterable[ArtifactContainer], run: wandb_run
) -> None:
    allowed_artifact_types = [Path, pd.DataFrame]

    for artifact_container in artifact_containers:
        if artifact_container.artifact not in allowed_artifact_types:
            raise TypeError(
                f"Type of artifact is {type(artifact_container.artifact)}, must be one of {allowed_artifact_types}"
            )

        if isinstance(artifact_container.artifact, Path):
            log_image_to_wandb(
                chart_path=artifact_container.artifact,
                chart_name=artifact_container.label,
                run=run,
            )
        elif isinstance(artifact_container.artifact, pd.DataFrame):
            wandb_table = wandb.Table(dataframe=artifact_container.artifact)
            run.log_artifact({artifact_container.label: wandb_table})


def run_full_evaluation(
    cfg: FullConfig,
    eval_dataset: EvalDataset,
    save_dir: Path,
    run: wandb_run,
    pipe_metadata: Optional[PipeMetadata] = None,
):
    df = pd.DataFrame()

    eval_dataset = eval_dataset(
        ids=df["dw_ek_borger"],
        pred_timestamps=df["pred_timestamps"],
        outcome_timestamps=df["outcome_timestamps"],
        y=df["y"],
        y_hat_probs=df["y_hat_prob"],
        y_hat_int=df["y_hat_int"],
    )

    lookahead_bins, lookbehind_bins = filter_plot_bins(cfg=cfg)

    artifact_containers = add_base_plots(
        cfg=cfg,
        eval_dataset=eval_dataset,
        lookahead_bins=lookahead_bins,
        lookbehind_bins=lookbehind_bins,
        save_dir=save_dir,
    )

    if pipe_metadata and pipe_metadata.feature_importances:
        artifact_containers += [
            ArtifactContainer(
                label="feature_importances",
                artifact=plot_feature_importances(
                    feature_importance_dict=pipe_metadata.feature_importances,
                    save_path=save_dir / "feature_importances.png",
                ),
            ),
            ArtifactContainer(
                label="feature_importances",
                artifact=generate_feature_importances_table(
                    feature_importance_dict=pipe_metadata.feature_importances,
                    output_format="df",
                ),
            ),
        ]

    upload_artifacts(run=run, artifact_containers=artifact_containers)


def filter_plot_bins(
    cfg: FullConfig,
):
    # Bins for plotting
    lookahead_bins: Iterable[int] = cfg.eval.lookahead_bins
    lookbehind_bins: Iterable[int] = cfg.eval.lookbehind_bins

    # Drop date_bins_direction if they are further away than min_lookdirection_days
    if cfg.data.min_lookbehind_days:
        lookbehind_bins = [
            b for b in lookbehind_bins if cfg.data.min_lookbehind_days < b
        ]

    if cfg.data.min_lookahead_days:
        lookahead_bins = [
            b for b in lookahead_bins if cfg.data.min_lookahead_days < abs(b)
        ]

    # Invert date_bins_behind to negative if it's not already
    if min(lookbehind_bins) >= 0:
        lookbehind_bins = [-d for d in lookbehind_bins]

    return lookahead_bins, lookbehind_bins


def add_base_plots(
    cfg,
    eval_dataset,
    save_dir,
    lookahead_bins: Sequence[Union[int, float]],
    lookbehind_bins: Sequence[Union[int, float]],
):
    pred_proba_percentiles = positive_rate_to_pred_probs(
        pred_probs=eval_dataset.y_hat_probs,
        positive_rate_thresholds=cfg.eval.positive_rate_thresholds,
    )

    return [
        ArtifactContainer(
            label="sensitivity_by_time_by_threshold",
            artifact=plot_sensitivity_by_time_to_outcome_heatmap(
                eval_dataset=eval_dataset,
                pred_proba_thresholds=pred_proba_percentiles,
                bins=lookbehind_bins,
                save_path=save_dir / "sensitivity_by_time_by_threshold.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_time_from_first_visit",
            artifact=plot_auc_by_time_from_first_visit(
                eval_dataset=eval_dataset,
                bins=lookbehind_bins,
                save_path=save_dir / "auc_by_time_from_first_visit.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_calendar_time",
            artifact=plot_metric_by_calendar_time(
                eval_dataset=eval_dataset,
                y_title="AUC",
                save_path=save_dir / "auc_by_calendar_time.png",
            ),
        ),
        ArtifactContainer(
            label="recall_by_time_to_diagnosis",
            artifact=plot_metric_by_time_until_diagnosis(
                eval_dataset=eval_dataset,
                bins=lookahead_bins,
                metric_fn=recall_score,
                y_title="Sensitivity (recall)",
                save_path=save_dir / "recall_by_time_to_diagnosis.png",
            ),
        ),
        ArtifactContainer(
            label="performance_by_threshold",
            artifact=generate_performance_by_positive_rate_table(
                eval_dataset=eval_dataset,
                pred_proba_thresholds=pred_proba_percentiles,
                positive_rate_thresholds=cfg.eval.positive_rate_thresholds,
                output_format="df",
            ),
        ),
    ]
