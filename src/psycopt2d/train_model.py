"""Training script for training a single model for predicting t2d."""

from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import wandb

# import wandb
from pandas import Series
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from psycopt2d.evaluate_model import evaluate_model
from psycopt2d.feature_transformers import ConvertToBoolean, DateTimeConverter
from psycopt2d.load import load_dataset
from psycopt2d.models import model_catalogue
from psycopt2d.utils import flatten_nested_dict

CONFIG_PATH = Path(__file__).parent / "config"
TRAINING_COL_NAME_PREFIX = "pred_"


def create_preprocessing_pipeline(cfg):
    """create preprocessing pipeline based on config."""
    steps = []

    if cfg.preprocessing.convert_datetimes_to:
        dtconverter = DateTimeConverter(
            convert_to=cfg.preprocessing.convert_datetimes_to,
        )
        steps.append(("DateTimeConverter", dtconverter))

    if cfg.preprocessing.convert_to_boolean:
        steps.append(("ConvertToBoolean", ConvertToBoolean()))

    return Pipeline(steps)


def create_model(cfg):
    model_config_dict = model_catalogue.get(cfg.model.model_name)

    model_args = model_config_dict["static_hyperparameters"]

    training_arguments = getattr(cfg.model, cfg.model.model_name)
    model_args.update(training_arguments)

    mdl = model_config_dict["model"](**model_args)
    return mdl


def train_with_predefined_splits(cfg, OUTCOME_COL_NAME, pipe) -> Tuple[Series, Series]:
    """Loads dataset and fits a model on the pre-defined split.

    Args:
        cfg (_type_): _description_
        OUTCOME_COL_NAME (_type_): _description_

    Returns:
        Tuple(Series, Series): Two series: True labels and predicted labels for the validation set.
    """
    X_train, y_train, val, X_val, y_val = load_predefined_splits(cfg, OUTCOME_COL_NAME)

    pipe.fit(X_train, y_train)
    y_train_hat = pipe.predict_proba(X_train)[:, 1]
    y_val_hat = pipe.predict_proba(X_val)[:, 1]

    print(f"Performance on train: {round(roc_auc_score(y_train, y_train_hat), 3)}")

    return val, y_val, y_val_hat


def load_predefined_splits(cfg, OUTCOME_COL_NAME):
    # Train set
    train = load_dataset(
        split_names="train",
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
    )
    X_train = train[
        [c for c in train.columns if c.startswith(cfg.data.pred_col_name_prefix)]
    ]
    y_train = train[OUTCOME_COL_NAME]

    # Val set
    val = load_dataset(
        split_names="val",
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
    )

    X_val = val[[c for c in val.columns if c.startswith(cfg.data.pred_col_name_prefix)]]
    y_val = val[[OUTCOME_COL_NAME]]
    return X_train, y_train, val, X_val, y_val


def train_with_crossvalidation(cfg, OUTCOME_COL_NAME, pipe):
    """Loads train and val, concatenates them and uses them for cross-
    validation.

    Args:
        cfg (_type_): _description_
        OUTCOME_COL_NAME (_type_): _description_

    Returns:
        Tuple(Series, Series): Two series: True labels and predicted labels for the validation set.
    """
    dataset = load_dataset(
        split_names=["train", "val"],
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
    )

    # Get predictors
    X = dataset[
        [c for c in dataset.columns if c.startswith(cfg.data.pred_col_name_prefix)]
    ]
    y = dataset[[OUTCOME_COL_NAME]]

    # Create folds
    folds = StratifiedGroupKFold(n_splits=cfg.training.n_splits).split(
        X,
        y,
        dataset["dw_ek_borger"],
    )

    # Perform CV and get out of fold predictions
    dataset["oof_y_hat"] = np.nan
    for train_idxs, val_idxs in folds:
        X_, y_ = X.loc[train_idxs], y.loc[train_idxs]
        pipe.fit(X_, y_)

        y_hat = pipe.predict_proba(X_)[:, 1]
        print(f"Within-fold performance: {round(roc_auc_score(y_,y_hat), 3)}")

        out_of_fold_y = dataset[OUTCOME_COL_NAME].loc[val_idxs]
        dataset["oof_y_hat"].loc[val_idxs] = pipe.predict_proba(X.loc[val_idxs])[:, 1]

        out_of_fold_y_hat = dataset["oof_y_hat"].loc[val_idxs]
        print(
            f"Out-of-fold performance: {round(roc_auc_score(out_of_fold_y, out_of_fold_y_hat), 3)}",
        )

    return (
        dataset[
            [c for c in dataset.columns if c != OUTCOME_COL_NAME and c != "oof_y_hat"]
        ],
        dataset[OUTCOME_COL_NAME],
        dataset["oof_y_hat"],
    )


def train_with_synth_data(cfg, OUTCOME_COL_NAME, pipe):
    """Loads train and val, concatenates them and uses them for cross-
    validation.

    Args:
        cfg (_type_): _description_
        OUTCOME_COL_NAME (_type_): _description_

    Returns:
        Tuple(Series, Series): Two series: True labels and predicted labels for the validation set.
    """

    # Get top_directory in package
    base_dir = Path(__file__).parent.parent.parent

    dataset = pd.read_csv(
        base_dir / "tests" / "test_data" / "synth_prediction_data.csv",
    )

    # Get 75% of dataset for train
    train = dataset.sample(frac=0.75, random_state=42)
    X_train = train[
        [c for c in train.columns if c.startswith(cfg.data.pred_col_name_prefix)]
    ]
    y_train = train[OUTCOME_COL_NAME]

    # Get remaining 25% for val
    val = dataset.drop(train.index)
    X_val = val[[c for c in val.columns if c.startswith(cfg.data.pred_col_name_prefix)]]
    y_val = val[[OUTCOME_COL_NAME]]

    pipe.fit(X_train, y_train)
    y_val_hat = pipe.predict_proba(X_val)[:, 1]

    return (
        val[[c for c in dataset.columns if c != OUTCOME_COL_NAME and c != "oof_y_hat"]],
        y_val,
        y_val_hat,
    )


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="train_config",
)
def main(cfg):
    if cfg.evaluation.wandb:
        run = wandb.init(
            project=cfg.project.name,
            reinit=True,
            config=flatten_nested_dict(cfg, sep="."),
        )
    else:
        run = None

    OUTCOME_COL_NAME = (
        f"outc_dichotomous_t2d_within_{cfg.data.lookahead_days}_days_max_fallback_0"
    )

    preprocessing_pipe = create_preprocessing_pipeline(cfg)
    mdl = create_model(cfg)

    if len(preprocessing_pipe.steps) != 0:
        pipe = Pipeline([("preprocessing", preprocessing_pipe), ("mdl", mdl)])
    else:
        pipe = Pipeline([("mdl", mdl)])

    if cfg.training.data_source == "synth":
        OUTCOME_COL_NAME = "outc_dichotomous_t2d_within_30_days_max_fallback_0"
        X_eval, y, y_hat_prob = train_with_synth_data(cfg, OUTCOME_COL_NAME, pipe)
    elif cfg.training.n_splits is not None:
        X_eval, y, y_hat_prob = train_with_crossvalidation(cfg, OUTCOME_COL_NAME, pipe)
    else:
        X_eval, y, y_hat_prob = train_with_predefined_splits(
            cfg,
            OUTCOME_COL_NAME,
            pipe,
        )

    # Calculate performance metrics and log to wandb_run
    evaluate_model(X=X_eval, y=y, y_hat_prob=y_hat_prob, run=run, cfg=cfg)

    # finish run
    if cfg.evaluation.wandb:
        run.finish()


if __name__ == "__main__":
    main()
