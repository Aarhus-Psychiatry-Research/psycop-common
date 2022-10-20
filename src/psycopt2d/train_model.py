"""Training script for training a single model for predicting t2d."""
import os
from collections.abc import Iterable
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import wandb
from omegaconf.dictconfig import DictConfig
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wasabi import Printer

from psycopt2d.evaluation import evaluate_model
from psycopt2d.feature_transformers import ConvertToBoolean, DateTimeConverter
from psycopt2d.load import load_train_and_val_from_cfg
from psycopt2d.models import MODELS
from psycopt2d.utils import (
    PROJECT_ROOT,
    create_wandb_folders,
    flatten_nested_dict,
    get_feature_importance_dict,
    prediction_df_with_metadata_to_disk,
)

CONFIG_PATH = PROJECT_ROOT / "src" / "psycopt2d" / "config"
TRAINING_COL_NAME_PREFIX = "pred_"

# Handle wandb not playing nice with joblib
os.environ["WANDB_START_METHOD"] = "thread"


def create_preprocessing_pipeline(cfg):
    """Create preprocessing pipeline based on config."""
    steps = []

    if cfg.preprocessing.convert_datetimes_to:
        dtconverter = DateTimeConverter(
            convert_to=cfg.preprocessing.convert_datetimes_to,
        )
        steps.append(("DateTimeConverter", dtconverter))

    if cfg.preprocessing.convert_to_boolean:
        steps.append(("ConvertToBoolean", ConvertToBoolean()))

    if cfg.model.require_imputation:
        steps.append(
            ("Imputation", SimpleImputer(strategy=cfg.preprocessing.imputation_method)),
        )
    if cfg.preprocessing.transform in {
        "z-score-normalization",
        "z-score-normalisation",
    }:
        steps.append(
            ("z-score-normalization", StandardScaler()),
        )

    return Pipeline(steps)


def create_model(cfg):
    """Instantiate and return a model object based on settings in the config
    file."""
    model_dict = MODELS.get(cfg.model.model_name)

    model_args = model_dict["static_hyperparameters"]

    training_arguments = getattr(cfg.model, "args")
    model_args.update(training_arguments)

    mdl = model_dict["model"](**model_args)
    return mdl


def stratified_cross_validation(
    cfg,
    pipe: Pipeline,
    train_df: pd.DataFrame,
    train_col_names: Iterable[str],
    outcome_col_name: str,
    n_splits: int,
):
    """Performs stratified and grouped cross validation using the pipeline."""
    msg = Printer(timestamp=True)

    X = train_df[train_col_names]  # pylint: disable=invalid-name
    y = train_df[outcome_col_name]  # pylint: disable=invalid-name

    # Create folds
    msg.info("Creating folds")
    folds = StratifiedGroupKFold(n_splits=n_splits).split(
        X=X,
        y=y,
        groups=train_df[cfg.data.id_col_name],
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


def train_and_eval_on_crossvalidation(
    cfg: DictConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: Iterable[str],
    n_splits: int,
) -> pd.DataFrame:
    """Train model on cross validation folds and return evaluation dataset.

    Args:
        cfg (DictConfig): Config object
        train: Training dataset
        val: Validation dataset
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training
        n_splits: Number of folds for cross validation.

    Returns:
        Evaluation dataset
    """
    msg = Printer(timestamp=True)

    msg.info("Concatenating train and val for crossvalidation")
    train_val = pd.concat([train, val], ignore_index=True)

    eval_dataset = stratified_cross_validation(
        cfg=cfg,
        pipe=pipe,
        train_df=train_val,
        train_col_names=train_col_names,
        outcome_col_name=outcome_col_name,
        n_splits=n_splits,
    )

    eval_dataset.rename(columns={"oof_y_hat": "y_hat_prob"}, inplace=True)
    return eval_dataset


def train_and_eval_on_val_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
) -> pd.DataFrame:
    """Train model on pre-defined train and validation split and return
    evaluation dataset.

    Args:
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

    eval_dataset = val
    eval_dataset["y_hat_prob"] = y_val_hat_prob
    return eval_dataset


def train_and_get_model_eval_df(
    cfg: DictConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
    n_splits: Optional[int],
) -> pd.DataFrame:
    """Train model and return evaluation dataset.

    Args:
        cfg (DictConfig): Config object
        train: Training dataset
        val: Validation dataset
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training
        n_splits: Number of folds for cross validation. If None, no cross validation is performed.

    Returns:
        Evaluation dataset
    """
    # Set feature names if model is EBM to get interpretable feature importance
    # output
    if cfg.model.model_name in ("ebm", "xgboost"):
        pipe["model"].feature_names = train_col_names

    if n_splits is None:  # train on pre-defined splits
        eval_dataset = train_and_eval_on_val_split(
            train=train,
            val=val,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
        )
    else:
        eval_dataset = train_and_eval_on_crossvalidation(
            cfg=cfg,
            train=train,
            val=val,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
            n_splits=n_splits,
        )

    return eval_dataset


def create_pipeline(cfg):
    """Create pipeline.

    Args:
        cfg (DictConfig): Config object

    Returns:
        Pipeline
    """
    steps = []
    preprocessing_pipe = create_preprocessing_pipeline(cfg)
    if len(preprocessing_pipe.steps) != 0:
        steps.append(("preprocessing", preprocessing_pipe))

    mdl = create_model(cfg)
    steps.append(("model", mdl))
    pipe = Pipeline(steps)
    return pipe


def get_col_names(cfg: DictConfig, train: pd.DataFrame) -> tuple[str, list[str]]:
    """Get column names for outcome and features.

    Args:
        cfg (DictConfig): Config object
        train: Training dataset

    Returns:
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training
    """

    outcome_col_name = (  # pylint: disable=invalid-name
        f"outc_dichotomous_t2d_within_{cfg.data.lookahead_days}_days_max_fallback_0"
    )

    train_col_names = [  # pylint: disable=invalid-name
        c for c in train.columns if c.startswith(cfg.data.pred_col_name_prefix)
    ]

    return outcome_col_name, train_col_names


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(cfg):
    """Main function for training a single model."""
    msg = Printer(timestamp=True)

    create_wandb_folders()

    run = wandb.init(
        project=cfg.project.name,
        reinit=True,
        config=flatten_nested_dict(cfg, sep="."),
        mode=cfg.project.wandb_mode,
        group=cfg.project.wandb_group,
    )

    dataset = load_train_and_val_from_cfg(cfg)

    msg.info("Creating pipeline")
    pipe = create_pipeline(cfg)

    outcome_col_name, train_col_names = get_col_names(cfg, dataset.train)

    msg.info("Training model")
    eval_df = train_and_get_model_eval_df(
        cfg=cfg,
        train=dataset.train,
        val=dataset.val,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
        n_splits=cfg.training.n_splits,
    )

    # Save model predictions, feature importance, and config to disk
    prediction_df_with_metadata_to_disk(df=eval_df, cfg=cfg, pipe=pipe, run=run)

    # only run full evaluation if wandb mode mode is online
    # otherwise delegate to watcher script
    if cfg.project.wandb_mode == "run":
        msg.info("Evaluating model")
        evaluate_model(
            cfg=cfg,
            eval_df=eval_df,
            y_col_name=outcome_col_name,
            y_hat_prob_col_name="y_hat_prob",
            feature_importance_dict=get_feature_importance_dict(pipe),
            run=run,
        )

    roc_auc = roc_auc_score(
        eval_df[outcome_col_name],
        eval_df["y_hat_prob"],
    )

    msg.info(f"ROC AUC: {roc_auc}")
    run.log({"roc_auc_unweighted": roc_auc})
    run.finish()
    return roc_auc


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
