"""Train a single model and evaluate it."""
from pathlib import Path
from typing import Optional

import wandb
from psycop_model_training.application_modules.wandb_handler import WandbHandler
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop_model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop_model_training.training.train_and_predict import train_and_predict
from psycop_model_training.training_output.model_evaluator import ModelEvaluator
from psycop_model_training.utils.col_name_inference import get_col_names
from psycop_model_training.utils.decorators import (
    wandb_alert_on_exception_return_terrible_auc,
)
from psycop_model_training.utils.utils import PROJECT_ROOT, SHARED_RESOURCES_PATH


def get_eval_dir(cfg: FullConfigSchema) -> Path:
    """Get the directory to save evaluation results to."""
    # If online
    ovartaci_path = (
        SHARED_RESOURCES_PATH
        / cfg.project.name
        / "model_eval"
        / wandb.run.group  # type: ignore
        / wandb.run.name  # type: ignore
    )

    if cfg.project.wandb.group == "integration_testing":
        eval_dir_path = PROJECT_ROOT / "tests" / "test_eval_results"
    else:
        eval_dir_path = ovartaci_path

    eval_dir_path.mkdir(parents=True, exist_ok=True)

    return eval_dir_path


@wandb_alert_on_exception_return_terrible_auc
def post_wandb_setup_train_model(
    cfg: FullConfigSchema,
    override_output_dir: Optional[Path] = None,
) -> float:
    """Train a single model and evaluate it."""
    eval_dir_path = get_eval_dir(cfg)

    train_datasets = [
        load_and_filter_split_from_cfg(
            data_cfg=cfg.data,
            pre_split_cfg=cfg.preprocessing.pre_split,
            split=split,
        )
        for split in cfg.data.splits_for_training
    ]

    if cfg.data.splits_for_evaluation is not None:
        eval_datasets = [
            load_and_filter_split_from_cfg(
                data_cfg=cfg.data,
                pre_split_cfg=cfg.preprocessing.pre_split,
                split=split,  # type: ignore
            )
            for split in cfg.data.splits_for_evaluation
        ]
    else:
        eval_datasets = None

    pipe = create_post_split_pipeline(cfg)
    outcome_col_name, train_col_names = get_col_names(cfg, train_datasets[0])

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
