"""Train a single model and evaluate it."""
from typing import Callable, Optional

import wandb

from psycop_model_training.application_modules.wandb_handler import WandbHandler
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.utils import (
    load_and_filter_train_and_val_from_cfg,
)
from psycop_model_training.model_eval.dataclasses import ArtifactContainer
from psycop_model_training.model_eval.model_evaluator import ModelEvaluator
from psycop_model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop_model_training.training.train_and_predict import train_and_predict
from psycop_model_training.utils.col_name_inference import get_col_names
from psycop_model_training.utils.utils import PROJECT_ROOT, SHARED_RESOURCES_PATH


def train_model(cfg: FullConfigSchema, custom_artifact_fn: Optional[Callable] = None):
    """Main function for training a single model."""
    WandbHandler(cfg=cfg).setup_wandb()
    eval_dir_path = get_eval_dir(cfg)

    dataset = load_and_filter_train_and_val_from_cfg(cfg)
    pipe = create_post_split_pipeline(cfg)
    outcome_col_name, train_col_names = get_col_names(cfg, dataset.train)

    eval_dataset = train_and_predict(
        cfg=cfg,
        train=dataset.train,
        val=dataset.val,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
        n_splits=cfg.train.n_splits,
    )

    if custom_artifact_fn:
        custom_artifacts = custom_artifact_fn(
            eval_dataset=eval_dataset,
            save_dir=eval_dir_path,
        )

    roc_auc = ModelEvaluator(
        eval_dir_path=eval_dir_path,
        cfg=cfg,
        pipe=pipe,
        eval_ds=eval_dataset,
        raw_train_set=dataset.train,
        custom_artifacts=custom_artifacts,
        upload_to_wandb=cfg.project.wandb.mode != "offline",
    ).evaluate()

    return roc_auc


def get_eval_dir(cfg: FullConfigSchema):
    """Get the directory to save evaluation results to."""
    if wandb.run.id and cfg.project.wandb.mode != "offline":
        eval_dir_path = SHARED_RESOURCES_PATH / cfg.project.name / wandb.run.name
    else:
        eval_dir_path = PROJECT_ROOT / "tests" / "test_eval_results"
        eval_dir_path.mkdir(parents=True, exist_ok=True)

    return eval_dir_path
