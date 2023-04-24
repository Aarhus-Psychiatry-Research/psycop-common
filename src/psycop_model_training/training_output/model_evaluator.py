import logging
from pathlib import Path

# Set matplotlib backend to Agg to avoid errors when running on a server in parallel
import pandas as pd
import wandb
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.training_output.artifact_saver.to_disk import (
    ArtifactsToDiskSaver,
)
from psycop_model_training.training_output.dataclasses import (
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

        if hasattr(self.pipe["model"], "feature_importances_"):  # type: ignore
            pipe_metadata.feature_importances = get_feature_importance_dict(
                pipe=self.pipe,
            )

        if "preprocessing" in self.pipe and hasattr(
            self.pipe["preprocessing"].named_steps,  # type: ignore
            "feature_selection",
        ):
            pipe_metadata.selected_features = get_selected_features_dict(
                pipe=self.pipe,
                train_col_names=self.train_col_names,
            )

        return pipe_metadata

    def __init__(
        self,
        cfg: FullConfigSchema,
        eval_dir_path: Path,
        raw_train_set: pd.DataFrame,
        pipe: Pipeline,
        eval_ds: EvalDataset,
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

        self.pipeline_metadata = self._get_pipeline_metadata()
        self.disk_saver = ArtifactsToDiskSaver(dir_path=eval_dir_path)

    def evaluate_and_save_eval_data(self) -> float:
        """Evaluate the model and save artifacts."""
        roc_auc: float = roc_auc_score(  # type: ignore
            self.eval_ds.y,
            self.eval_ds.y_hat_probs,
        )

        self.disk_saver.save(
            cfg=self.cfg,
            eval_dataset=self.eval_ds,
            pipe=self.pipe,
            pipe_metadata=self.pipeline_metadata,
            roc_auc=roc_auc,
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

        logging.info(
            f"ROC AUC: {roc_auc}",
        )

        return roc_auc
