"""Experiments with models trained and evaluated using only part of the data"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from wasabi import Printer

from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.common.model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop.common.model_training.training.utils import create_eval_dataset
from psycop.common.model_training.utils.col_name_inference import get_col_names
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
    RunGroup,
)


def train_validate(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
) -> EvalDataset:
    """Train model on pre-defined train and validation split and return
    evaluation dataset.

    Args:
        cfg (FullConfig): Config object
        train: Training dataset
        val: Validation dataset
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training

    Returns:
        Evaluation dataset
    """

    X_train = train[train_col_names]
    y_train = train[outcome_col_name]
    X_val = val[train_col_names]

    pipe.fit(X_train, y_train)

    y_train_hat_prob = pipe.predict_proba(X_train)[:, 1]
    y_val_hat_prob = pipe.predict_proba(X_val)[:, 1]

    print(
        f"Performance on train: {round(roc_auc_score(y_train, y_train_hat_prob), 3)}"  # type: ignore
    )

    df = val
    df["y_hat_prob"] = y_val_hat_prob

    return create_eval_dataset(
        col_names=cfg.data.col_name, outcome_col_name=outcome_col_name, df=df
    )


def load_data(cfg: FullConfigSchema) -> tuple(pd.DataFrame, pd.DataFrame): # type: ignore
    """Train a single model and evaluate it."""
    train_dataset = pd.concat(
        [
            load_and_filter_split_from_cfg(
                data_cfg=cfg.data, pre_split_cfg=cfg.preprocessing.pre_split, split=split
            )
            for split in cfg.data.splits_for_training
        ],
        ignore_index=True,
    )   

    if cfg.data.splits_for_evaluation is not None:

        val_dataset = pd.concat(
            [
                load_and_filter_split_from_cfg(
                    data_cfg=cfg.data, pre_split_cfg=cfg.preprocessing.pre_split, split=split
                )
                for split in cfg.data.splits_for_evaluation
            ],
            ignore_index=True,
        )

    else: 
        val_dataset = load_and_filter_split_from_cfg(
                    data_cfg=cfg.data, pre_split_cfg=cfg.preprocessing.pre_split, split='val'
                )

    return train_dataset, val_dataset


def train_model_on_different_training_data_amounts(cfg: FullConfigSchema) -> tuple[float, list[float], list[float]]:
    """Train a single model and evaluate it."""
    train_dataset, val_dataset = load_data(cfg)

    outcome_col_name_for_train, train_col_names = get_col_names(cfg, train_dataset)

    pipe = create_post_split_pipeline(cfg)

    eval_dataset = train_validate(
        cfg=cfg,
        train=train_dataset,
        val=val_dataset,
        pipe=pipe,
        outcome_col_name=outcome_col_name_for_train,  # type: ignore
        train_col_names=train_col_names,
    )

    roc_auc = roc_auc_score(  # type: ignore
        eval_dataset.y, eval_dataset.y_hat_probs
    )
    return roc_auc  # type: ignore


def plot_performance_by_amount_of_training_data(
    models_to_train: pd.DataFrame, models_descriptions: str | None = None
) -> pd.DataFrame:
    return pd.DataFrame()