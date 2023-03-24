import logging
from collections.abc import Sequence
from pathlib import Path, PosixPath, WindowsPath

import matplotlib as mpl

# Set matplotlib backend to Agg to avoid errors when running on a server in parallel
mpl.use("Agg")
import pandas as pd
import wandb
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.model_eval.artifact_saver.to_disk import ArtifactsToDiskSaver
from psycop_model_training.model_eval.base_artifacts.plots.utils import (
    log_image_to_wandb,
)
from psycop_model_training.model_eval.dataclasses import (
    ArtifactContainer,
    EvalDataset,
    PipeMetadata,
)
from psycop_model_training.utils.col_name_inference import get_col_names
from psycop_model_training.utils.utils import (
    get_feature_importance_dict,
    get_selected_features_dict,
)
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating a model."""

    def _get_pipeline_metadata(self) -> PipeMetadata:
        pipe_metadata = PipeMetadata()

        if hasattr(self.pipe["model"], "feature_importances_"):
            pipe_metadata.feature_importances = get_feature_importance_dict(
                pipe=self.pipe,
            )

        if "preprocessing" in self.pipe and hasattr(
            self.pipe["preprocessing"].named_steps,
            "feature_selection",
        ):
            pipe_metadata.selected_features = get_selected_features_dict(
                pipe=self.pipe,
                train_col_names=self.train_col_names,
            )

        return pipe_metadata

    def __init__(
        self,
        eval_dir_path: Path,
        cfg: FullConfigSchema,
        raw_train_set: pd.DataFrame,
        pipe: Pipeline,
        artifacts: Sequence[ArtifactContainer],
        eval_ds: EvalDataset,
        upload_to_wandb: bool = True,
    ):
        """Class for evaluating a model.

        Args:
            eval_dir_path (Path): Path to directory where artifacts will be saved.
            cfg (FullConfigSchema): Full config object.
            artifacts (Sequence[ArtifactContainer]): List of artifacts to save.
            raw_train_set (pd.DataFrame): Training set before feature selection.
            pipe (Pipeline): Pipeline object.
            eval_ds (EvalDataset): EvalDataset object.
            upload_to_wandb (bool, optional): Whether to upload artifacts to wandb. Defaults to True.
        """
        self.cfg = cfg
        self.pipe = pipe
        self.eval_ds = eval_ds
        self.outcome_col_name, self.train_col_names = get_col_names(
            cfg,
            dataset=raw_train_set,
        )

        self.eval_dir_path = eval_dir_path
        self.pipeline_metadata = self._get_pipeline_metadata()

        self.disk_saver = ArtifactsToDiskSaver(dir_path=eval_dir_path)
        self.artifacts = artifacts
        self.upload_to_wandb = upload_to_wandb

    def upload_artifact_to_wandb(
        self,
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

    def evaluate(self) -> float:
        """Evaluate the model and save artifacts."""
        self.disk_saver.save(
            cfg=self.cfg,
            eval_dataset=self.eval_ds,
            pipe=self.pipe,
            pipe_metadata=self.pipeline_metadata,
        )

        roc_auc = roc_auc_score(
            self.eval_ds.y,
            self.eval_ds.y_hat_probs,
        )

        wandb.log(
            {
                "roc_auc_unweighted": roc_auc,
                "lookbehind": max(
                    self.cfg.preprocessing.pre_split.lookbehind_combination,
                ),
                "lookahead": self.cfg.preprocessing.pre_split.min_lookahead_days,
            },
        )

        logging.info(  # pylint: disable=logging-not-lazy,logging-fstring-interpolation
            f"ROC AUC: {roc_auc}",
        )

        if self.upload_to_wandb and self.artifacts is not None:
            for artifact in self.artifacts:
                self.upload_artifact_to_wandb(artifact)

        return roc_auc
