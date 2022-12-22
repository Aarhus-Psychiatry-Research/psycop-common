"""Train a single model and evaluate it."""
import wandb
from omegaconf import DictConfig, OmegaConf
from wasabi import Printer

from application.artifacts.custom_artifacts import create_custom_plot_artifacts
from psycop_model_training.data_loader.utils import (
    load_and_filter_train_and_val_from_cfg,
)
from psycop_model_training.model_eval.dataclasses import PipeMetadata
from psycop_model_training.model_eval.evaluate_model import evaluate_performance
from psycop_model_training.model_eval.model_evaluator import ModelEvaluator
from psycop_model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop_model_training.training.train_and_predict import train_and_predict
from psycop_model_training.utils.col_name_inference import get_col_names
from psycop_model_training.utils.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema
from psycop_model_training.utils.utils import PROJECT_ROOT, SHARED_RESOURCES_PATH


def train_model(cfg: DictConfig):
    """Main function for training a single model."""
    if not isinstance(cfg, FullConfigSchema):
        cfg = convert_omegaconf_to_pydantic_object(cfg)

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

    eval_dir_path = SHARED_RESOURCES_PATH / cfg.project.name / wandb.run.id

    custom_artifacts = create_custom_plot_artifacts(
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
    ).evaluate()

    return roc_auc
