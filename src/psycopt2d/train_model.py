"""Training script for training a single model for predicting t2d."""
import os
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import pandas as pd
import wandb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from psycopt2d.evaluate_model import evaluate_model
from psycopt2d.feature_transformers import ConvertToBoolean, DateTimeConverter
from psycopt2d.load import load_dataset
from psycopt2d.models import MODELS
from psycopt2d.utils import flatten_nested_dict

CONFIG_PATH = Path(__file__).parent / "config"
TRAINING_COL_NAME_PREFIX = "pred_"

# Handle wandb not playing nice with joblib
os.environ["WANDB_START_METHOD"] = "thread"


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
    model_dict = MODELS.get(cfg.model.model_name)

    model_args = model_dict["static_hyperparameters"]

    training_arguments = getattr(cfg.model, "args")
    model_args.update(training_arguments)

    mdl = model_dict["model"](**model_args)
    return mdl


def load_dataset_from_config(cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """load dataset based on config file.

    Should only use the cfg.data config with the exception of seed
    """

    if cfg.data.source.lower() == "sql":
        train = load_dataset(
            split_names="train",
            table_name=cfg.data.table_name,
            n_training_samples=cfg.data.n_training_samples,
            drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
            min_lookahead_days=cfg.data.min_lookahead_days,
        )
        val = load_dataset(
            split_names="val",
            table_name=cfg.data.table_name,
            n_training_samples=cfg.data.n_training_samples,
            drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
            min_lookahead_days=cfg.data.min_lookahead_days,
        )
    elif cfg.data.source.lower() == "synthetic":
        repo_dir = Path(__file__).parent.parent.parent
        dataset = pd.read_csv(
            repo_dir / "tests" / "test_data" / "synth_prediction_data.csv",
        )

        # Convert all timestamp cols to datetime
        for col in [col for col in dataset.columns if "timestamp" in col]:
            dataset[col] = pd.to_datetime(dataset[col])

        # Get 75% of dataset for train
        train, val = train_test_split(
            dataset,
            test_size=0.25,
            random_state=cfg.project.seed,
        )
    else:
        raise ValueError(
            "The config data.source is {cfg.data.source}",
        )
    return train, val


def stratified_cross_validation(
    cfg,
    pipe: Pipeline,
    dataset: pd.DataFrame,
    train_col_names: List[str],
    outcome_col_name: str,
):
    """Performs a stratified and grouped cross validation using the
    pipeline."""
    X = dataset[train_col_names]
    y = dataset[outcome_col_name]

    # Create folds
    folds = StratifiedGroupKFold(n_splits=cfg.training.n_splits).split(
        X=X,
        y=y,
        groups=dataset[cfg.data.id_col_name],
    )

    # Perform CV and get out of fold predictions
    dataset["oof_y_hat"] = np.nan
    for train_idxs, val_idxs in folds:
        X_, y_ = X.loc[train_idxs], y.loc[train_idxs]
        pipe.fit(X_, y_)

        y_hat = pipe.predict_proba(X_)[:, 1]
        print(f"Within-fold performance: {round(roc_auc_score(y_,y_hat), 3)}")
        dataset["oof_y_hat_prob"].loc[val_idxs] = pipe.predict_proba(X.loc[val_idxs])[
            :,
            1,
        ]

    return dataset


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="default_config",
    version_base="1.2",
)
def main(cfg):
    run = wandb.init(
        project=cfg.project.name,
        reinit=True,
        config=flatten_nested_dict(cfg, sep="."),
        mode=cfg.project.wandb_mode,
    )

    # load dataset
    train, val = load_dataset_from_config(cfg)

    # creating pipeline
    steps = []
    preprocessing_pipe = create_preprocessing_pipeline(cfg)
    if len(preprocessing_pipe.steps) != 0:
        steps.append(("preprocessing", preprocessing_pipe))

    mdl = create_model(cfg)
    steps.append(("model", mdl))
    pipe = Pipeline(steps)

    # train
    ## define columns
    OUTCOME_COL_NAME = (
        f"outc_dichotomous_t2d_within_{cfg.data.lookahead_days}_days_max_fallback_0"
    )
    if cfg.data.source.lower() == "synthetic":
        OUTCOME_COL_NAME = "outc_dichotomous_t2d_within_30_days_max_fallback_0"

    TRAIN_COL_NAMES = [
        c for c in train.columns if c.startswith(cfg.data.pred_col_name_prefix)
    ]

    # Set feature names if model is EBM to get interpretable feature importance
    # output
    if cfg.model.model_name == "ebm":
        pipe["model"].feature_names = TRAIN_COL_NAMES

    if cfg.training.n_splits is None:  # train on pre-defined splits
        X_train = train[TRAIN_COL_NAMES]
        y_train = train[OUTCOME_COL_NAME]
        X_val = val[TRAIN_COL_NAMES]

        pipe.fit(X_train, y_train)

        y_train_hat_prob = pipe.predict_proba(X_train)[:, 1]
        y_val_hat_prob = pipe.predict_proba(X_val)[:, 1]

        print(
            f"Performance on train: {round(roc_auc_score(y_train, y_train_hat_prob), 3)}",
        )  # TODO log to wandb

        eval_dataset = val
        eval_dataset["y_hat_prob"] = y_val_hat_prob
        y_hat_prob_col_name = "y_hat_prob"
    else:
        train_val = pd.concat([train, val], ignore_index=True)
        eval_dataset = stratified_cross_validation(
            cfg,
            pipe,
            dataset=train_val,
            train_col_names=TRAIN_COL_NAMES,
            outcome_col_name=OUTCOME_COL_NAME,
        )
        y_hat_prob_col_name = "oof_y_hat_prob"

    # Evaluate: Calculate performance metrics and log to wandb
    evaluate_model(
        cfg=cfg,
        pipe=pipe,
        eval_dataset=eval_dataset,
        y_col_name=OUTCOME_COL_NAME,
        train_col_names=TRAIN_COL_NAMES,
        y_hat_prob_col_name=y_hat_prob_col_name,
        run=run,
    )

    run.finish()

    return roc_auc_score(
        eval_dataset[OUTCOME_COL_NAME],
        eval_dataset[y_hat_prob_col_name],
    )


if __name__ == "__main__":
    main()
