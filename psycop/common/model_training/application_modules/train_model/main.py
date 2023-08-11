"""Train a single model and evaluate it."""
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb

from psycop.common.global_utils.paths import PSYCOP_PKG_ROOT
from psycop.common.model_training.application_modules.wandb_handler import WandbHandler
from psycop.common.model_training.config_schemas.conf_utils import (
    validate_classification_objective,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.common.model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop.common.model_training.training.train_and_predict import train_and_predict
from psycop.common.model_training.training_output.model_evaluator import ModelEvaluator
from psycop.common.model_training.utils.col_name_inference import get_col_names
from psycop.common.model_training.utils.decorators import (
    wandb_alert_on_exception_return_terrible_auc,
)


def get_eval_dir(cfg: FullConfigSchema) -> Path:
    """Get the directory to save evaluation results to."""
    if cfg.project.wandb.group == "integration_testing":
        eval_dir_path = PSYCOP_PKG_ROOT / "model_training" / "test_eval_results"
    else:
        eval_dir_path = (
            cfg.project.project_path
            / "pipeline_eval"
            / wandb.run.group  # type: ignore
            / wandb.run.name  # type: ignore
        )

    eval_dir_path.mkdir(parents=True, exist_ok=True)
    return eval_dir_path


@wandb_alert_on_exception_return_terrible_auc
def post_wandb_setup_train_model(
    cfg: FullConfigSchema,
    override_output_dir: Optional[Path] = None,
) -> float:
    """Train a single model and evaluate it."""
    eval_dir_path = get_eval_dir(cfg)

    train_datasets = pd.concat(
        [
            load_and_filter_split_from_cfg(
                data_cfg=cfg.data,
                pre_split_cfg=cfg.preprocessing.pre_split,
                split=split,
            )
            for split in cfg.data.splits_for_training
        ],
        ignore_index=True,
    )

    if cfg.data.splits_for_evaluation is not None:
        eval_datasets = pd.concat(
            [
                load_and_filter_split_from_cfg(
                    data_cfg=cfg.data,
                    pre_split_cfg=cfg.preprocessing.pre_split,
                    split=split,  # type: ignore
                )
                for split in cfg.data.splits_for_evaluation
            ],
            ignore_index=True,
        )
    else:
        eval_datasets = None

    pipe = create_post_split_pipeline(cfg)
    outcome_col_name, train_col_names = get_col_names(cfg, train_datasets)
    validate_classification_objective(cfg=cfg, col_names=outcome_col_name)

    eval_dataset = train_and_predict(
        cfg=cfg,
        train_datasets=train_datasets,
        val_datasets=eval_datasets,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
    )

    eval_dir = eval_dir_path if override_output_dir is None else override_output_dir

    roc_auc = ModelEvaluator(
        eval_dir_path=eval_dir,
        cfg=cfg,
        pipe=pipe,
        eval_ds=eval_dataset,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
    ).evaluate_and_save_eval_data()

    return roc_auc


def train_model(
    cfg: FullConfigSchema,
    override_output_dir: Optional[Path] = None,
) -> float:
    """Main function for training a single model."""
    WandbHandler(cfg=cfg).setup_wandb()

    # Try except block ensures process doesn't die in the case of an exception,
    # but rather logs to wandb and starts another run with a new combination of
    # hyperparameters
    roc_auc = post_wandb_setup_train_model(cfg, override_output_dir=override_output_dir)

    return roc_auc
