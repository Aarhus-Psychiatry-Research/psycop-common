"""Training script for training a single model."""
import os
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from wasabi import Printer

from psycop.common.global_utils.paths import PSYCOP_PKG_ROOT
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.training.model_specs import MODELS
from psycop.common.model_training.training.utils import create_eval_dataset
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.model_training.utils.utils import SUPPORTS_MULTILABEL_CLASSIFICATION

CONFIG_PATH = PSYCOP_PKG_ROOT / "application" / "config"

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

    if cfg.preprocessing.pre_split.classification_objective == "multilabel":
        return MultiOutputClassifier(model_dict["model"](**model_args))

    return model_dict["model"](**model_args)


def stratified_cross_validation(
    cfg: FullConfigSchema,
    pipe: Pipeline,
    train_df: pd.DataFrame,
    train_col_names: list[str],
    outcome_col_name: str,
) -> pd.DataFrame:
    """Performs stratified and grouped cross validation using the pipeline."""
    msg = Printer(timestamp=True)

    X = train_df[train_col_names]
    y = train_df[outcome_col_name]

    # Create folds
    msg.info("Creating folds")
    msg.info(f"Training on {X.shape[1]} columns and {X.shape[0]} rows")

    folds = StratifiedGroupKFold(n_splits=cfg.n_crossval_splits).split(
        X=X,
        y=y,
        groups=train_df[cfg.data.col_name.id],
    )

    # Perform CV and get out of fold predictions
    train_df["oof_y_hat"] = np.nan

    for i, (train_idxs, val_idxs) in enumerate(folds):
        msg_prefix = f"Fold {i + 1}"

        msg.info(f"{msg_prefix}: Training fold")

        X_train, y_train = (
            X.loc[train_idxs],
            y.loc[train_idxs],
        )
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X_train)[:, 1]

        msg.info(f"{msg_prefix}: Train AUC = {round(roc_auc_score(y_train,y_pred), 3)}")  # type: ignore

        oof_y_pred = pipe.predict_proba(X.loc[val_idxs])[  # type: ignore
            :,
            1,
        ]

        msg.info(f"{msg_prefix}: Oof AUC = {round(roc_auc_score(y.loc[val_idxs],oof_y_pred), 3)}")  # type: ignore

        train_df.loc[val_idxs, "oof_y_hat"] = oof_y_pred

    return train_df


def multilabel_cross_validation(
    cfg: FullConfigSchema,
    pipe: Pipeline,
    train_df: pd.DataFrame,
    train_col_names: list[str],
    outcome_col_name: list[str],
) -> pd.DataFrame:
    """Performs stratified and grouped cross validation using the pipeline."""
    if cfg.model.name not in SUPPORTS_MULTILABEL_CLASSIFICATION:
        raise ValueError(
            f"{cfg.model.name} does not support multilabel classification. Models that support multilabel classification include: {SUPPORTS_MULTILABEL_CLASSIFICATION}.",
        )

    msg = Printer(timestamp=True)

    X = train_df[train_col_names]
    y = train_df[outcome_col_name]

    # Create folds
    msg.info("Creating folds")
    msg.info(f"Training on {X.shape[1]} columns and {X.shape[0]} rows")

    # StratifiedGroupKFold does not support stratification across multiple labels, so we create a concatenated label that represents all individual labels
    y_joined = y.astype(int).astype(str).apply("".join, axis=1)

    # stratify by first outcome column
    folds = StratifiedGroupKFold(n_splits=cfg.n_crossval_splits).split(
        X=X,
        y=y_joined,
        groups=train_df[cfg.data.col_name.id],
    )

    msg.info(
        f"Stratifying by {y[outcome_col_name[0]]}",  # type: ignore
    )

    # Perform CV and get out of fold predictions
    train_df["oof_y_hat"] = np.nan

    for i, (train_idxs, val_idxs) in enumerate(folds):
        msg_prefix = f"Fold {i + 1}"

        msg.info(f"{msg_prefix}: Training fold")

        X_train, y_train = (
            X.loc[train_idxs],
            y.loc[train_idxs],
        )

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X_train)

        msg.info(
            f"{msg_prefix}: Train AUC = {round(roc_auc_score(y_train, np.asarray([y_pred[x][:,1] for x in range(0, len(y_pred))]).T), 3)}",  # type: ignore
        )

        oof_y_pred = pipe.predict_proba(X.loc[val_idxs])

        msg.info(
            f"{msg_prefix}: Oof AUC = {round(roc_auc_score(y.loc[val_idxs], np.asarray([oof_y_pred[x][:,1] for x in range(0, len(oof_y_pred))]).T), 3)}",  # type: ignore
        )

        train_df.loc[
            val_idxs,
            [f"y_hat_prob_{x}" for x in outcome_col_name],
        ] = np.asarray([oof_y_pred[x][:, 1] for x in range(0, len(oof_y_pred))]).T

    return train_df


def crossvalidate(
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


def multilabel_crossvalidate(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: list[str],
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

    df = multilabel_cross_validation(
        cfg=cfg,
        pipe=pipe,
        train_df=train,
        train_col_names=train_col_names,
        outcome_col_name=outcome_col_name,
    )

    return create_eval_dataset(
        col_names=cfg.data.col_name,
        outcome_col_name=outcome_col_name,
        df=df,
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
        f"Performance on train: {round(roc_auc_score(y_train, y_train_hat_prob), 3)}",  # type: ignore
    )

    df = val
    df["y_hat_prob"] = y_val_hat_prob

    return create_eval_dataset(
        col_names=cfg.data.col_name,
        outcome_col_name=outcome_col_name,
        df=df,
    )


def multilabel_train_validate(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: list[str],
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

    y_train_hat_prob = pipe.predict_proba(X_train)
    y_val_hat_prob = pipe.predict_proba(X_val)

    print(
        f"Performance on train: {round(roc_auc_score(y_train, np.asarray([y_train_hat_prob[x][:,1] for x in range(0, len(y_train_hat_prob))]).T), 3)}",  # type: ignore
    )

    df = val
    df.loc[
        :,
        [f"y_hat_prob_{x}" for x in outcome_col_name],
    ] = np.asarray([y_val_hat_prob[x][:, 1] for x in range(0, len(y_val_hat_prob))]).T

    return create_eval_dataset(
        col_names=cfg.data.col_name,
        outcome_col_name=outcome_col_name,
        df=df,
    )


def train_and_predict(
    cfg: FullConfigSchema,
    train_datasets: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: Union[str, list[str]],
    train_col_names: list[str],
    val_datasets: Optional[pd.DataFrame] = None,
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

    if cfg.preprocessing.pre_split.classification_objective == "binary":
        if val_datasets is not None:  # train on pre-defined splits
            eval_dataset = train_validate(
                cfg=cfg,
                train=train_datasets,
                val=val_datasets,
                pipe=pipe,
                outcome_col_name=outcome_col_name,  # type: ignore
                train_col_names=train_col_names,
            )
        else:
            eval_dataset = crossvalidate(
                cfg=cfg,
                train=train_datasets,
                pipe=pipe,
                outcome_col_name=outcome_col_name,  # type: ignore
                train_col_names=train_col_names,
            )

    elif cfg.preprocessing.pre_split.classification_objective == "multilabel":
        if val_datasets is not None:
            eval_dataset = multilabel_train_validate(
                cfg=cfg,
                train=train_datasets,
                val=val_datasets,
                pipe=pipe,
                outcome_col_name=outcome_col_name,  # type: ignore
                train_col_names=train_col_names,
            )
        else:
            eval_dataset = multilabel_crossvalidate(
                cfg=cfg,
                train=train_datasets,
                pipe=pipe,
                outcome_col_name=outcome_col_name,  # type: ignore
                train_col_names=train_col_names,
            )

    return eval_dataset  # type: ignore
