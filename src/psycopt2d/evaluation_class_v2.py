"""_summary_"""
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import wandb
from omegaconf import DictConfig
from pydantic import BaseModel
from sklearn.metrics import recall_score
from wandb.sdk.wandb_run import Run as wandb_run  # pylint: disable=no-name-in-module

from psycopt2d.tables.performance_by_threshold import (
    generate_performance_by_positive_rate_table,
)
from psycopt2d.utils import positive_rate_to_pred_probs
from psycopt2d.visualization.performance_over_time import (
    plot_auc_by_time_from_first_visit,
    plot_metric_by_calendar_time,
    plot_metric_by_time_until_diagnosis,
)
from psycopt2d.visualization.sens_over_time import (
    plot_sensitivity_by_time_to_outcome_heatmap,
)
from psycopt2d.visualization.utils import log_image_to_wandb


class EvalDataset(BaseModel):
    ids: pd.Series
    pred_timestamps: pd.Series
    outcome_timestamps: pd.Series
    y: pd.Series
    y_hat_probs: pd.Series
    y_hat_int: pd.Series


class ArtifactSpecification(BaseModel):
    label: str
    artifact_generator_fn: Callable[[Any], Union[pd.DataFrame, Path]]
    kwargs: dict


class ArtifactContainer(BaseModel):
    label: str
    artifact: Union[Path, pd.DataFrame]


class PipeMetadata(BaseModel):
    feature_importances: dict[str, float]


class ModelEvaluator:
    def __init__(self, eval_dataset: EvalDataset):
        self.eval_dataset = eval_dataset
        self.artifact_containers = []

    def add_artifact(
        self,
        artifact_spec: ArtifactSpecification,
        kwargs: Optional[dict] = None,
    ):
        artifact = artifact_spec.artifact_generator_fn(
            eval_dataset=self.eval_dataset, **kwargs
        )
        self.artifact_containers.append(
            ArtifactContainer(label=artifact_spec.label, artifact=artifact),
        )

    def upload_artifacts(self, run: wandb_run):
        for artifact in self.artifact_containers:
            if isinstance(artifact.artifact, Path):
                log_image_to_wandb(
                    chart_path=artifact.artifact,
                    chart_name=artifact.label,
                    run=run,
                )
            elif isinstance(artifact.artifact, pd.DataFrame):
                wandb_table = wandb.Table(dataframe=artifact.artifact)
                run.log_artifact({artifact.label: wandb_table})
            else:
                raise ValueError("Artifact type is ")


def run_full_evaluation(
    cfg: DictConfig,
    eval_dataset: EvalDataset,
    artifact_specification: ArtifactSpecification,
    save_dir: Path,
    run: wandb_run,
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

    pred_proba_thresholds = positive_rate_to_pred_probs(
        pred_probs=eval_dataset.y_hat_probs,
        positive_rate_thresholds=cfg.evaluation.positive_rate_thresholds,
    )

    date_bins_ahead: Iterable[int] = cfg.evaluation.date_bins_ahead
    date_bins_behind: Iterable[int] = cfg.evaluation.date_bins_behind

    # Drop date_bins_direction if they are further away than min_lookdirection_days
    if cfg.data.min_lookbehind_days:
        date_bins_behind = [
            b for b in date_bins_behind if cfg.data.min_lookbehind_days < b
        ]

    if cfg.data.min_lookahead_days:
        date_bins_ahead = [
            b for b in date_bins_ahead if cfg.data.min_lookahead_days < abs(b)
        ]

    # Invert date_bins_behind to negative if it's not already
    if min(date_bins_behind) >= 0:
        date_bins_behind = [-d for d in date_bins_behind]

    evaluator = ModelEvaluator(eval_dataset=eval_dataset)

    artifact_specifications = (
        artifact_specification(
            label="sensitivity_by_time_by_threshold",
            artifact_generator_fn=plot_sensitivity_by_time_to_outcome_heatmap,
        ),
        artifact_specification(
            label="auc_by_time_from_first_visit",
            artifact_generator_fn=plot_auc_by_time_from_first_visit,
        ),
        artifact_specification(
            label="auc_by_calendar_time",
            artifact_generator_fn=plot_metric_by_calendar_time,
        ),
        artifact_specification(
            label="recall_by_time_to_diagnosis",
            artifact_generator_fn=plot_metric_by_time_until_diagnosis,
            kwargs={"metric": recall_score, "y_title": "Sensitivity (recall)"},
        ),
        artifact_specification(
            label="performance_by_threshold",
            artifact_generator_fn=generate_performance_by_positive_rate_table,
            kwargs={
                "pred_proba_thresholds": pred_proba_thresholds,
                "positive_rate_thresholds": cfg.evaluation.positive_rate_thresholds,
                "output_format": "df",
            },
        ),
    )

    for artifact_spec in artifact_specifications:
        artifact_spec.kwargs["save_path"] = save_dir / artifact_spec.name
        evaluator.add_artifact(artifact_spec=artifact_spec)

    evaluator.upload_artifacts(run=run)
