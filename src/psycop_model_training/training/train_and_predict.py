"""Training script for training a single model."""
import os
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.training.model_specs import MODELS
from psycop_model_training.training.utils import create_eval_dataset
from psycop_model_training.training_output.dataclasses import EvalDataset
from psycop_model_training.utils.utils import PROJECT_ROOT
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from wasabi import Printer

CONFIG_PATH = PROJECT_ROOT / "application" / "config"

# Handle wandb not playing nice with joblib
os.environ["WANDB_START_METHOD"] = "thread"
log = Printer(timestamp=True)


def create_model(cfg: FullConfigSchema) -> Any:
    """Instantiate and return a model object based on settings in the config
    file."""
    model_dict: dict[str, Any] = MODELS.get(cfg.model.name)  # type: ignore

    model_args = model_dict["static_hyperparameters"]

    training_arguments = cfg.model.args
    model_args.update(training_arguments)

    return model_dict["model"](**model_args)


def stratified_cross_validation(  # pylint: disable=too-many-locals
    cfg: FullConfigSchema,
    pipe: Pipeline,
    train_df: pd.DataFrame,
    train_col_names: list[str],
    outcome_col_name: str,
) -> pd.DataFrame:
    """Performs stratified and grouped cross validation using the pipeline."""
    msg = Printer(timestamp=True)

    X = train_df[train_col_names]  # pylint: disable=invalid-name
    y = train_df[outcome_col_name]  # pylint: disable=invalid-name

    # Create folds
    msg.info("Creating folds")
    msg.info(f"Training on {X.shape[1]} columns and {X.shape[0]} rows")

    folds = StratifiedGroupKFold(n_splits=5).split(
        X=X,
        y=y,
        groups=train_df[cfg.data.col_name.id],
    )

    # Perform CV and get out of fold predictions
    train_df["oof_y_hat"] = np.nan

    for i, (train_idxs, val_idxs) in enumerate(folds):
        msg_prefix = f"Fold {i + 1}"

        msg.info(f"{msg_prefix}: Training fold")

        X_train, y_train = (  # pylint: disable=invalid-name
            X.loc[train_idxs],
            y.loc[train_idxs],
        )  # pylint: disable=invalid-name
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X_train)[:, 1]

        msg.info(f"{msg_prefix}: AUC = {round(roc_auc_score(y_train,y_pred), 3)}")

        train_df.loc[val_idxs, "oof_y_hat"] = pipe.predict_proba(X.loc[val_idxs])[
            :,
            1,
        ]

    return train_df


def crossval_train_and_predict(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
) -> EvalDataset:
    """Train model on cross validation folds and return evaluation dataset.

    Args:
        cfg: Config object
        train: Training dataset
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training

    Returns:
        Evaluation dataset
    """

    df = stratified_cross_validation(
        cfg=cfg,
        pipe=pipe,
        train_df=train,
        train_col_names=train_col_names,
        outcome_col_name=outcome_col_name,
    )

    df = df.rename(columns={"oof_y_hat": "y_hat_prob"})

    return create_eval_dataset(
        col_names=cfg.data.col_name,
        outcome_col_name=outcome_col_name,
        df=df,
    )


def train_val_predict(
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

    X_train = train[train_col_names]  # pylint: disable=invalid-name
    y_train = train[outcome_col_name]
    X_val = val[train_col_names]  # pylint: disable=invalid-name

    pipe.fit(X_train, y_train)

    y_train_hat_prob = pipe.predict_proba(X_train)[:, 1]
    y_val_hat_prob = pipe.predict_proba(X_val)[:, 1]

    print(
        f"Performance on train: {round(roc_auc_score(y_train, y_train_hat_prob), 3)}",
    )

    df = val
    df["y_hat_prob"] = y_val_hat_prob

    return create_eval_dataset(
        col_names=cfg.data.col_name,
        outcome_col_name=outcome_col_name,
        df=df,
    )


def train_and_predict(
    cfg: FullConfigSchema,
    train_datasets: Sequence[pd.DataFrame],
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
    val_datasets: Optional[Sequence[pd.DataFrame]] = None,
) -> EvalDataset:
    """Train model and return evaluation dataset.

    Args:
        cfg: Config object
        train_datasets: Training datasets
        val_datasets: Validation datasets
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training

    Returns:
        Evaluation dataset
    """
    # Set feature names if model is EBM to get interpretable feature importance
    # output
    log.good("Training model")
    if cfg.model.name in ("ebm", "xgboost"):
        pipe["model"].feature_names = train_col_names  # type: ignore

    train = pd.concat(train_datasets, ignore_index=True)

    if val_datasets is not None:  # train on pre-defined splits
        val = pd.concat(val_datasets, ignore_index=True)
        eval_dataset = train_val_predict(
            cfg=cfg,
            train=train,
            val=val,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
        )
    else:
        eval_dataset = crossval_train_and_predict(
            cfg=cfg,
            train=train,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
        )

    return eval_dataset
