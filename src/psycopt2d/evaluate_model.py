"""_summary_"""
from collections.abc import Iterable
from pathlib import Path, PosixPath, WindowsPath
from typing import Optional

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
from psycopt2d.tables.tables import generate_feature_importances_table, generate_selected_features_table
from psycopt2d.utils.config_schemas import FullConfigSchema
from psycopt2d.utils.utils import positive_rate_to_pred_probs
from psycopt2d.visualization.feature_importance import plot_feature_importances
from psycopt2d.visualization.performance_by_age import plot_performance_by_age
from psycopt2d.visualization.performance_by_n_hba1c import plot_performance_by_n_hba1c
from psycopt2d.visualization.performance_over_time import (
    plot_auc_by_time_from_first_visit,
    plot_metric_by_calendar_time,
    plot_metric_by_cyclic_time,
    plot_metric_by_time_until_diagnosis,
)
from psycopt2d.visualization.roc_auc import plot_auc_roc
from psycopt2d.visualization.sens_over_time import (
    plot_sensitivity_by_time_to_outcome_heatmap,
)
from psycopt2d.visualization.utils import log_image_to_wandb


def upload_artifacts_to_wandb(
    artifact_containers: Iterable[ArtifactContainer],
    run: wandb_run,
) -> None:
    """Upload artifacts to wandb."""
    allowed_artifact_types = [Path, WindowsPath, PosixPath, pd.DataFrame]

    for artifact_container in artifact_containers:
        if type(artifact_container.artifact) not in allowed_artifact_types:
            raise TypeError(
                f"Type of artifact is {type(artifact_container.artifact)}, must be one of {allowed_artifact_types}",
            )

        if isinstance(artifact_container.artifact, Path):
            log_image_to_wandb(
                chart_path=artifact_container.artifact,
                chart_name=artifact_container.label,
                run=run,
            )
        elif isinstance(artifact_container.artifact, pd.DataFrame):
            wandb_table = wandb.Table(dataframe=artifact_container.artifact)
            run.log({artifact_container.label: wandb_table})


def create_base_plot_artifacts(
    cfg: FullConfigSchema,
    eval_dataset: EvalDataset,
    save_dir: Path,
) -> list[ArtifactContainer]:
    """A collection of plots that are always generated."""
    pred_proba_percentiles = positive_rate_to_pred_probs(
        pred_probs=eval_dataset.y_hat_probs,
        positive_rate_thresholds=cfg.eval.positive_rate_thresholds,
    )

    lookahead_bins = cfg.eval.lookahead_bins

    return [
        ArtifactContainer(
            label="sensitivity_by_time_by_threshold",
            artifact=plot_sensitivity_by_time_to_outcome_heatmap(
                eval_dataset=eval_dataset,
                pred_proba_thresholds=pred_proba_percentiles,
                bins=lookahead_bins,
                save_path=save_dir / "sensitivity_by_time_by_threshold.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_time_from_first_visit",
            artifact=plot_auc_by_time_from_first_visit(
                eval_dataset=eval_dataset,
                bins=lookahead_bins,
                save_path=save_dir / "auc_by_time_from_first_visit.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_calendar_time",
            artifact=plot_metric_by_calendar_time(
                eval_dataset=eval_dataset,
                save_path=save_dir / "auc_by_calendar_time.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_hour_of_day",
            artifact=plot_metric_by_cyclic_time(
                eval_dataset=eval_dataset,
                bin_period="H",
                save_path=save_dir / "auc_by_calendar_time.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_day_of_week",
            artifact=plot_metric_by_cyclic_time(
                eval_dataset=eval_dataset,
                bin_period="D",
                save_path=save_dir / "auc_by_calendar_time.png",
            ),
        ),
        ArtifactContainer(
            label="auc_by_month_of_year",
            artifact=plot_metric_by_cyclic_time(
                eval_dataset=eval_dataset,
                bin_period="M",
                save_path=save_dir / "auc_by_calendar_time.png",
            ),
        ),
        ArtifactContainer(
            label="auc_roc",
            artifact=plot_auc_roc(
                eval_dataset=eval_dataset,
                save_path=save_dir / "auc_roc.png",
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
        ArtifactContainer(
            label="performance_by_age",
            artifact=plot_performance_by_age(
                eval_dataset=eval_dataset,
                save_path=save_dir / "performance_by_age.png",
            ),
        ),
    ]


def create_custom_plot_artifacts(
    eval_dataset: EvalDataset,
    save_dir: Path,
) -> list[ArtifactContainer]:
    """A collection of plots that are always generated."""
    return [
        ArtifactContainer(
            label="performance_by_age",
            artifact=plot_performance_by_n_hba1c(
                eval_dataset=eval_dataset,
                save_path=save_dir / "performance_by_age.png",
            ),
        ),
    ]


def run_full_evaluation(
    cfg: FullConfigSchema,
    eval_dataset: EvalDataset,
    save_dir: Path,
    pipe_metadata: Optional[PipeMetadata] = None,
    run: Optional[wandb_run] = None,
    upload_to_wandb: bool = True,
):
    """Run the full evaluation. Uploads to wandb if upload_to_wandb is true.

    Args:
        cfg: The config for the evaluation.
        eval_dataset: The dataset to evaluate.
        save_dir: The directory to save plots to.
        run: The wandb run to upload to. Determines whether to upload to wandb.
        pipe_metadata: The metadata for the pipe.
        upload_to_wandb: Whether to upload to wandb.
    """

    # Create the directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    artifact_containers = create_base_plot_artifacts(
        cfg=cfg,
        eval_dataset=eval_dataset,
        save_dir=save_dir,
    )

    artifact_containers += create_custom_plot_artifacts(
        eval_dataset=eval_dataset,
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

    if pipe_metadata and pipe_metadata.selected_features:
        artifact_containers += [
            ArtifactContainer(
                label="selected_features",
                artifact=generate_selected_features_table(
                    selected_features_dict=pipe_metadata.selected_features,
                    output_format="df",
                ),
            ),
        ]

    if upload_to_wandb and run is None:
        raise ValueError("Must pass a run to be able to upload to wandb.")

    if upload_to_wandb and run:
        upload_artifacts_to_wandb(run=run, artifact_containers=artifact_containers)
