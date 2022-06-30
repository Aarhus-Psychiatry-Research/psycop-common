"""Training script for training a single model for predicting t2d.

TODO:
add split for using pre-defined train-val split
add dynamic hyperparams for hydra optimisation

Features:
# fix impute
# move filter to compute
"""

from pathlib import Path
from typing import Iterable, Optional, Tuple

import hydra
import numpy as np
import wandb
from pandas import Series
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from wandb.sdk import wandb_run

from psycopt2d.feature_transformers import ConvertToBoolean, DateTimeConverter
from psycopt2d.load import load_dataset
from psycopt2d.models import MODELS
from psycopt2d.utils import flatten_nested_dict

CONFIG_PATH = Path(__file__).parent / "config"
TRAINING_COL_NAME_PREFIX = "pred_"


def create_preprocessing_pipelines(cfg):
    """create preprocessing pipeline based on config."""
    steps = []
    dtconverter = DateTimeConverter(convert_to=cfg.training.convert_datetimes_to)
    steps.append(("DateTimeConverter", dtconverter))

    if cfg.training.convert_to_boolean:
        steps.append(("ConvertToBoolean", ConvertToBoolean))
    return Pipeline(steps)


def create_model(cfg):
    model_args = MODELS[cfg.training.model_name]["static_hyperparameters"]
    training_arguments = cfg.getattr(cfg.training, cfg.training.model_name)
    model_args.update(training_arguments)
    mdl = MODELS[cfg.training.model_name](**model_args)
    return mdl


def evaluate(
    X,
    y: Iterable[int],
    y_hat_prob: Iterable[float],
    wandb_run: Optional[wandb_run.Run],
):
    if wandb_run:
        wandb_run.log({"roc_auc_unweighted": round(roc_auc_score(y, y_hat_prob), 3)})
    else:
        print(f"AUC is: {round(roc_auc_score(y, y_hat_prob), 3)}")
    # wandb.sklearn.plot_classifier(
    #     model,
    #     X_test=X,
    #     y_train=y_train,
    #     y_test=y_val,
    #     y_pred=y_preds,
    #     y_probas=y_probas,
    #     labels=[0, 1],
    #     model_name=cfg.training.model_name,
    #     feature_names=X_train.columns,
    # )
    # eval_df = X_val_eval
    # eval_df[OUTCOME_COL_NAME] = y_val_eval
    # eval_df[PREDICTED_OUTCOME_COL_NAME] = y_preds
    # eval_df[PREDICTED_PROBABILITY_COL_NAME] = y_probas

    # if cfg.evaluation.wandb:
    #     log_tpr_by_time_to_event(
    #         eval_df_combined=eval_df,
    #         outcome_col_name=OUTCOME_COL_NAME,
    #         predicted_outcome_col_name=PREDICTED_OUTCOME_COL_NAME,
    #         outcome_timestamp_col_name=OUTCOME_TIMESTAMP_COL_NAME,
    #         prediction_timestamp_col_name="timestamp",
    #         bins=[0, 1, 7, 14, 28, 182, 365, 730, 1825],
    #     )
    #     performance_metrics = calculate_performance_metrics(
    #         eval_df,
    #         outcome_col_name=OUTCOME_COL_NAME,
    #         prediction_probabilities_col_name=PREDICTED_PROBABILITY_COL_NAME,
    #         id_col_name="dw_ek_borger",
    #     )

    #     run.log(performance_metrics)


@hydra.main(
    config_path=CONFIG_PATH,
    config_name="train_config",
)
def main(cfg):
    if cfg.evaluation.wandb:
        run = wandb.init(
            project=cfg.project,
            reinit=True,
            config=flatten_nested_dict(cfg, sep="."),
        )
    else:
        run = None

    OUTCOME_COL_NAME = (
        f"outc_dichotomous_t2d_within_{cfg.data.lookahead_days}_days_max_fallback_0"
    )

    if cfg.training.n_splits is not None:
        y, y_hat_prob = cross_validated_performance(cfg, OUTCOME_COL_NAME)
    else:
        y, y_hat_prob = pre_defined_split_performance(cfg, OUTCOME_COL_NAME)

    # Calculate performance metrics and log to wandb_run
    evaluate(
        X="",
        y=y,
        y_hat_prob=y_hat_prob,
        wandb_run=run,
    )

    # finish run
    run.finish()


def pre_defined_split_performance(cfg, OUTCOME_COL_NAME) -> Tuple(Series, Series):
    """Loads dataset and fits a model on the pre-defined split.

    Args:
        cfg (_type_): _description_
        OUTCOME_COL_NAME (_type_): _description_

    Returns:
        Tuple(Series, Series): Two series: True labels and predicted labels for the validation set.
    """
    n_to_load = cfg.data.n_training_samples

    # Train set
    X_train, y_train = load_dataset(
        split_name="train",
        outcome_col_name=OUTCOME_COL_NAME,
        n_to_load=n_to_load,
    )
    X_train = X_train[
        [c for c in X_train.columns if c.startswith(cfg.data.pred_col_name_prefix)]
    ]

    # Val set
    X_val, y_val = load_dataset(
        split_name="val",
        n_to_load=n_to_load,
        outcome_col_name=OUTCOME_COL_NAME,
    )
    X_val = X_val[
        [c for c in X_val.columns if c.startswith(cfg.data.pred_col_name_prefix)]
    ]

    preprocessing_pipe = create_preprocessing_pipelines(cfg)
    mdl = create_model(cfg)
    pipe = Pipeline([("preprocessing", preprocessing_pipe), ("mdl", mdl)])

    pipe.fit(X_train, y_train)
    return y_val, pipe.predict(X_val)


def cross_validated_performance(cfg, OUTCOME_COL_NAME):
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
        drop_if_outcome_before_date=cfg.data.min_lookahead_days,
        min_lookahead_days=cfg.data.min_lookahead_days,
    )

    # extract training data
    X = dataset[
        [c for c in dataset.columns if c.startswith(cfg.data.pred_col_name_prefix)]
    ]
    y = dataset[[OUTCOME_COL_NAME]]

    # construct sklearn pipeline
    preprocessing_pipe = create_preprocessing_pipelines(cfg)
    mdl = create_model(cfg)
    pipe = Pipeline([("preprocessing", preprocessing_pipe), ("mdl", mdl)])

    # create stratified groups kfold
    folds = StratifiedGroupKFold(n_splits=cfg.training.n_splits).split(
        X,
        y,
        dataset["dw_ek_borger"],
    )

    # perform CV, obtaining out of fold predictions
    dataset["oof_y_hat"] = np.nan
    for train_idxs, val_idxs in folds:
        X_, y_ = X[train_idxs], y[train_idxs]
        pipe.fit(X_, y_)
        dataset["oof_y_hat"][val_idxs] = pipe.predict(X[val_idxs])

    return y, dataset["oof_y_hat"]


if __name__ == "__main__":
    main()
