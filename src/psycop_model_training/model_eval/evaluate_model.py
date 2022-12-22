"""_summary_"""
from pathlib import Path, PosixPath, WindowsPath
from typing import Optional

import pandas as pd
import wandb
from application.artifacts.custom_artifacts import create_custom_plot_artifacts

from psycop_model_training.model_eval.artifacts.base_plot_artifacts import create_base_plot_artifacts
from psycop_model_training.model_eval.dataclasses import (
    ArtifactContainer,
    EvalDataset,
    PipeMetadata,
)
from psycop_model_training.model_eval.artifacts.plots import (
    plot_feature_importances,
)
from psycop_model_training.model_eval.artifacts.plots import log_image_to_wandb
from psycop_model_training.model_eval.artifacts.tables import (
    generate_feature_importances_table,
    generate_selected_features_table,
)
from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema


def upload_artifact_to_wandb(
    artifact_container: ArtifactContainer,
) -> None:
    """Upload artifacts to wandb."""
    allowed_artifact_types = [Path, WindowsPath, PosixPath, pd.DataFrame]

    if type(artifact_container.artifact) not in allowed_artifact_types:
        raise TypeError(
            f"Type of artifact is {type(artifact_container.artifact)}, must be one of {allowed_artifact_types}",
        )

    if isinstance(artifact_container.artifact, Path):
        log_image_to_wandb(
            chart_path=artifact_container.artifact,
            chart_name=artifact_container.label,
        )
    elif isinstance(artifact_container.artifact, pd.DataFrame):
        wandb_table = wandb.Table(dataframe=artifact_container.artifact)
        wandb.log({artifact_container.label: wandb_table})


def evaluate_performance(
    cfg: FullConfigSchema,
    eval_dataset: EvalDataset,
    save_dir: Path,
    pipe_metadata: Optional[PipeMetadata] = None,
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


    if upload_to_wandb:
        for artifact_container in artifact_container:
            upload_artifact_to_wandb(artifact_container=artifact_container)
