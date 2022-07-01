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

# import wandb
from pandas import Series
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from wandb.sdk import wandb_run

import wandb
from psycopt2d.feature_transformers import ConvertToBoolean, DateTimeConverter
from psycopt2d.load import load_dataset
from psycopt2d.models import model_catalogue
from psycopt2d.utils import flatten_nested_dict

CONFIG_PATH = Path(__file__).parent / "config"
TRAINING_COL_NAME_PREFIX = "pred_"


def create_preprocessing_pipelines(cfg):
    """create preprocessing pipeline based on config."""
    steps = []
    dtconverter = DateTimeConverter(convert_to=cfg.preprocessing.convert_datetimes_to)
    steps.append(("DateTimeConverter", dtconverter))

    if cfg.preprocessing.convert_to_boolean:
        steps.append(("ConvertToBoolean", ConvertToBoolean()))

    # steps.append(("ConvertToNumpyArray", ConvertToNumpyArray()))
    return Pipeline(steps)


def create_model(cfg):
    model_config_dict = model_catalogue.get(cfg.model.model_name)

    model_args = model_config_dict["static_hyperparameters"]
    training_arguments = getattr(cfg.model, cfg.model.model_name)
    model_args.update(training_arguments)

    mdl = model_config_dict["model"](**model_args)
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

    preprocessing_pipe = create_preprocessing_pipelines(cfg)
    mdl = create_model(cfg)
    pipe = Pipeline([("mdl", mdl), ("preprocessing", preprocessing_pipe)])

    if cfg.training.n_splits is not None:
        y, y_hat_prob = cross_validated_performance(cfg, OUTCOME_COL_NAME, pipe)
    else:
        y, y_hat_prob = pre_defined_split_performance(cfg, OUTCOME_COL_NAME, pipe)

    # Calculate performance metrics and log to wandb_run
    evaluate(
        X="",
        y=y,
        y_hat_prob=y_hat_prob,
        wandb_run=run,
    )

    # finish run
    if cfg.evaluation.wandb:
        run.finish()


def pre_defined_split_performance(cfg, OUTCOME_COL_NAME, pipe) -> Tuple[Series, Series]:
    """Loads dataset and fits a model on the pre-defined split.

    Args:
        cfg (_type_): _description_
        OUTCOME_COL_NAME (_type_): _description_

    Returns:
        Tuple(Series, Series): Two series: True labels and predicted labels for the validation set.
    """
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

    pipe.fit(X_train, y_train)
    y_train_hat = pipe.predict_proba(X_train)[:, 1]
    y_val_hat = pipe.predict_proba(X_val)[:, 1]

    print(f"Performance on train: {round(roc_auc_score(y_train, y_train_hat), 3)}")

    return y_val, y_val_hat


def cross_validated_performance(cfg, OUTCOME_COL_NAME, pipe):
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

    # extract training data
    X = dataset[
        [c for c in dataset.columns if c.startswith(cfg.data.pred_col_name_prefix)]
    ]
    y = dataset[[OUTCOME_COL_NAME]]

    # create stratified groups kfold
    folds = StratifiedGroupKFold(n_splits=cfg.training.n_splits).split(
        X,
        y,
        dataset["dw_ek_borger"],
    )

    # perform CV, obtaining out of fold predictions
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

    return y, dataset["oof_y_hat"]


if __name__ == "__main__":
    main()
