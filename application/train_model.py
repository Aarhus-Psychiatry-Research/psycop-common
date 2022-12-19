import time
from typing import Any

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from wasabi import Printer

from psycop_model_training.data_loader.utils import (
    load_and_filter_train_and_val_from_cfg,
)
from psycop_model_training.model_eval.dataclasses import PipeMetadata
from psycop_model_training.model_eval.evaluate_model import run_full_evaluation
from psycop_model_training.training.train_and_eval import (
    CONFIG_PATH,
    train_and_get_model_eval_df,
)
from psycop_model_training.utils.col_name_inference import get_col_names
from psycop_model_training.utils.config_schemas.conf_utils import (
    convert_omegaconf_to_pydantic_object,
)
from psycop_model_training.utils.config_schemas.full_config import FullConfigSchema
from psycop_model_training.utils.utils import (
    PROJECT_ROOT,
    create_wandb_folders,
    eval_ds_cfg_pipe_to_disk,
    flatten_nested_dict,
    get_feature_importance_dict,
    get_selected_features_dict,
)
from psycop_model_training.preprocessing.post_split.pipeline import create_post_split_pipeline


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """Main function for training a single model."""
    # Save dictconfig for easier logging
    if isinstance(cfg, DictConfig):
        # Create flattened dict for logging to wandb
        # Wandb doesn't allow configs to be nested, so we
        # flatten it.
        dict_config_to_log: dict[str, Any] = flatten_nested_dict(OmegaConf.to_container(cfg), sep=".")  # type: ignore
    else:
        # For testing, we can take a FullConfig object instead. Simplifies boilerplate.
        dict_config_to_log = cfg.__dict__

    if not isinstance(cfg, FullConfigSchema):
        cfg = convert_omegaconf_to_pydantic_object(cfg)

    msg = Printer(timestamp=True)

    create_wandb_folders()

    run = wandb.init(
        project=cfg.project.name,
        reinit=True,
        config=dict_config_to_log,
        mode=cfg.project.wandb.mode,
        group=cfg.project.wandb.group,
        entity=cfg.project.wandb.entity,
    )

    if run is None:
        raise ValueError("Failed to initialise Wandb")

    # Add random delay based on cfg.train.random_delay_per_job to avoid
    # each job needing the same resources (GPU, disk, network) at the same time
    if cfg.train.random_delay_per_job_seconds:
        delay = np.random.randint(0, cfg.train.random_delay_per_job_seconds)
        msg.info(f"Delaying job by {delay} seconds to avoid resource competition")
        time.sleep(delay)

    dataset = load_and_filter_train_and_val_from_cfg(cfg)

    msg.info("Creating pipeline")
    pipe = create_post_split_pipeline(cfg)

    outcome_col_name, train_col_names = get_col_names(cfg, dataset.train)

    msg.info("Training model")
    eval_ds = train_and_get_model_eval_df(
        cfg=cfg,
        train=dataset.train,
        val=dataset.val,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
        n_splits=cfg.train.n_splits,
    )

    pipe_metadata = PipeMetadata()

    if hasattr(pipe["model"], "feature_importances_"):
        pipe_metadata.feature_importances = get_feature_importance_dict(pipe=pipe)
    if hasattr(pipe["preprocessing"].named_steps, "feature_selection"):
        pipe_metadata.selected_features = get_selected_features_dict(
            pipe=pipe,
            train_col_names=train_col_names,
        )

    # Save model predictions, feature importance, and config to disk
    eval_ds_cfg_pipe_to_disk(
        eval_dataset=eval_ds,
        cfg=cfg,
        pipe_metadata=pipe_metadata,
        run=run,
    )

    if cfg.project.wandb.mode == "run" or cfg.eval.force:
        msg.info("Evaluating model.")

        upload_to_wandb = cfg.project.wandb.mode == "run"

        run_full_evaluation(
            cfg=cfg,
            eval_dataset=eval_ds,
            run=run,
            pipe_metadata=pipe_metadata,
            save_dir=PROJECT_ROOT / "wandb" / "plots" / run.name,
            upload_to_wandb=upload_to_wandb,
        )

    roc_auc = roc_auc_score(
        eval_ds.y,
        eval_ds.y_hat_probs,
    )

    msg.info(f"ROC AUC: {roc_auc}")
    run.log(
        {
            "roc_auc_unweighted": roc_auc,
            "lookbehind": max(cfg.preprocessing.pre_split.lookbehind_combination),
            "lookahead": cfg.preprocessing.pre_split.min_lookahead_days,
        },
    )
    run.finish()
    return roc_auc


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
