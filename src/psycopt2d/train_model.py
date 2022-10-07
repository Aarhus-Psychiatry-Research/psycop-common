"""Training script for training a single model for predicting t2d."""
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import omegaconf
import pandas as pd
import wandb
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wasabi import Printer

from psycopt2d.evaluation import evaluate_model
from psycopt2d.feature_transformers import ConvertToBoolean, DateTimeConverter
from psycopt2d.load import load_dataset_from_dir
from psycopt2d.models import MODELS
from psycopt2d.utils import create_wandb_folders, flatten_nested_dict

CONFIG_PATH = Path(__file__).parent / "config"
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


def load_synth_train_val_from_dir(cfg, synth_splits_dir):
    """Load synthetic train and val data from dir."""
    # This is a temp fix. We probably want to use pydantic to validate all our inputs, and to set defaults if they don't exist in the config
    try:
        type(cfg.data.drop_patient_if_outcome_before_date)
    except omegaconf.errors.ConfigAttributeError:
        # Assign as none to struct
        with open_dict(cfg) as conf_dict:
            conf_dict.data.drop_patient_if_outcome_before_date = None

    train = load_dataset_from_dir(
        split_names="train",
        dir_path=synth_splits_dir,
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
        min_lookbehind_days=cfg.data.min_lookbehind_days,
        min_prediction_time_date=cfg.data.min_prediction_time_date,
        file_suffix="csv",
    )

    val = load_dataset_from_dir(
        split_names="val",
        dir_path=synth_splits_dir,
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
        min_lookbehind_days=cfg.data.min_lookbehind_days,
        min_prediction_time_date=cfg.data.min_prediction_time_date,
        file_suffix="csv",
    )

    return train, val


def gen_synth_data_splits(cfg, test_data_dir):
    """Generate synthetic data splits."""
    dataset = pd.read_csv(
        test_data_dir / "synth_prediction_data.csv",
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

    return train, val


def write_synth_splits(  # pylint: disable=unused-argument
    test_data_dir: Path,
    train,
    val,
):
    """Write synthetic data splits to disk."""

    synth_splits_dir = test_data_dir / "synth_splits"
    synth_splits_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        split_df = eval(split)  # pylint: disable=eval-used
        split_df.to_csv(synth_splits_dir / f"{split}.csv", index=False)

    return synth_splits_dir


def load_synthetic_data(cfg):
    """Load synthetic data from file."""
    repo_dir = Path(__file__).parent.parent.parent
    test_data_dir = repo_dir / "tests" / "test_data"

    train, val = gen_synth_data_splits(cfg, test_data_dir)

    synth_splits_dir = write_synth_splits(
        test_data_dir=test_data_dir,
        train=train,
        val=val,
    )

    # Load them from dir to use the same pipeline as we use for loading real data
    # Makes it actually act as a smoke test
    train, val = load_synth_train_val_from_dir(cfg, synth_splits_dir)

    return train, val


def load_real_data(cfg):
    """Load real data from file."""
    path = Path(cfg.data.dir)

    train = load_dataset_from_dir(
        split_names="train",
        dir_path=path,
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
        min_lookbehind_days=cfg.data.min_lookbehind_days,
    )

    val = load_dataset_from_dir(
        split_names="val",
        dir_path=path,
        n_training_samples=cfg.data.n_training_samples,
        drop_patient_if_outcome_before_date=cfg.data.drop_patient_if_outcome_before_date,
        min_lookahead_days=cfg.data.min_lookahead_days,
        min_lookbehind_days=cfg.data.min_lookbehind_days,
    )

    return train, val


def load_dataset_with_config(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset based on settings in the config file."""

    allowed_data_sources = {"csv", "parquet", "synthetic"}

    if "csv" in cfg.data.source.lower() or "parquet" in cfg.data.source.lower():
        train, val = load_real_data(cfg)

    elif cfg.data.source.lower() == "synthetic":
        train, val = load_synthetic_data(cfg)

    else:
        raise ValueError(
            f"The config data.source is {cfg.data.source}, allowed are {allowed_data_sources}",
        )

    return train, val


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
    train_col_names: Iterable[str],
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
    train_col_names: Iterable[str],
    n_splits: Optional[int],
):
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
    if cfg.model.model_name == "ebm":
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
    )

    train, val = load_dataset_with_config(cfg)

    msg.info("Creating pipeline")
    pipe = create_pipeline(cfg)

    outcome_col_name, train_col_names = get_col_names(cfg, train)

    msg.info("Training model")
    eval_df = train_and_get_model_eval_df(
        cfg=cfg,
        train=train,
        val=val,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
        n_splits=cfg.training.n_splits,
    )

    msg.info("Evaluating model")
    # Evaluate: Calculate performance metrics and log to wandb
    evaluate_model(
        cfg=cfg,
        pipe=pipe,
        eval_df=eval_df,
        y_col_name=outcome_col_name,
        train_col_names=train_col_names,
        y_hat_prob_col_name="y_hat_prob",
        run=run,
    )

    run.finish()

    roc_auc = roc_auc_score(
        eval_df[outcome_col_name],
        eval_df["y_hat_prob"],
    )

    msg.info(f"ROC AUC: {roc_auc}")

    return roc_auc


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
