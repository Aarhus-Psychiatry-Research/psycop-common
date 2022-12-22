import logging
from pathlib import Path

import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from application.artifacts.custom_artifacts import create_custom_plot_artifacts
from psycop_model_training.model_eval.artifact_saver.to_disk import ArtifactsToDiskSaver
from psycop_model_training.model_eval.artifacts.base_plot_artifacts import create_base_plot_artifacts
from psycop_model_training.model_eval.dataclasses import EvalDataset, PipeMetadata
from psycop_model_training.model_eval.evaluate_model import evaluate_performance
from psycop_model_training.utils.col_name_inference import get_col_names
from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema
from psycop_model_training.utils.utils import (
    get_feature_importance_dict,
    get_selected_features_dict,
)

log = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(
        self,
        eval_dir_path: Path,
        cfg: FullConfigSchema,
        raw_train_set: pd.DataFrame,
        pipe: Pipeline,
        eval_ds: EvalDataset,
        custom_artifacts: List[ArtifactContainer] = None,
    ):
        """Class for evaluating a model.

        Args:
            cfg (FullConfigSchema): Full config object.
            pipe (Pipeline): Pipeline object.
            eval_ds (EvalDataset): EvalDataset object.
            raw_train_set (pd.DataFrame): Training set before feature selection.
        """
        self.cfg = cfg
        self.pipe = pipe
        self.eval_ds = eval_ds
        self.outcome_col_name, self.train_col_names = get_col_names(
            cfg, dataset=raw_train_set
        )
        self.eval_dir_path = eval_dir_path
        self.disk_saver = ArtifactsToDiskSaver(run=wandb.run, dir_path=eval_dir_path)
        self.pipeline_metadata = self._get_pipeline_metadata()
        self.custom_artifacts = custom_artifacts

    def _get_pipeline_metadata(self):
        pipe_metadata = PipeMetadata()

        if hasattr(self.pipe["model"], "feature_importances_"):
            pipe_metadata.feature_importances = get_feature_importance_dict(
                pipe=self.pipe
            )
        if hasattr(self.pipe["preprocessing"].named_steps, "feature_selection"):
            pipe_metadata.selected_features = get_selected_features_dict(
                pipe=self.pipe,
                train_col_names=self.train_col_names,
            )

        return pipe_metadata

    def _create_artifacts(self) -> List[ArtifactContainer]:
        artifact_containers = create_base_plot_artifacts(
            cfg=self.cfg,
            eval_dataset=self.eval_dataset,
            save_dir=self.eval_dir_path,
        )

        if self.custom_artifacts:
            artifact_containers += self.custom_artifacts

        if pipe_metadata and pipe_metadata.feature_importances:
            artifact_containers += [
                ArtifactContainer(
                    label="feature_importances",
                    artifact=plot_feature_importances(
                        feature_importance_dict=self.pipe_metadata.feature_importances,
                        save_path=self.eval_dir_path / "feature_importances.png",
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

        return artifact_containers

    def evaluate(self) -> float:
        self.disk_saver.save(cfg=self.cfg, eval_dataset=self.eval_ds, pipe=self.pipe, pipe_metadata=self.pipeline_metadata)

        evaluate_performance(
            cfg=self.cfg,
            eval_dataset=self.eval_ds,
            save_dir_path=self.eval_dir_path,
            pipe_metadata=self.pipeline_metadata,
            upload_to_wandb=True,
        )

        roc_auc = roc_auc_score(
            self.eval_ds.y,
            self.eval_ds.y_hat_probs,
        )

        wandb.log(
            {
                "roc_auc_unweighted": roc_auc,
                "lookbehind": max(
                    self.cfg.preprocessing.pre_split.lookbehind_combination
                ),
                "lookahead": self.cfg.preprocessing.pre_split.min_lookahead_days,
            },
        )

        logging.info(f"ROC AUC: {roc_auc}")
