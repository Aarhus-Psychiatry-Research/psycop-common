"""Experiments with models trained and evaluated using different lookaheads for the outcome columns"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from wasabi import Printer

from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.utils import load_and_filter_split_from_cfg
from psycop.common.model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop.common.model_training.training.utils import create_eval_dataset
from psycop.common.model_training.utils.col_name_inference import get_col_names
from psycop.projects.forced_admission_outpatient.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
    RunGroup,
)


def _get_model_pipeline(
    model_name: str, group_name: str, model_type: int = 1
) -> ForcedAdmissionOutpatientPipelineRun:
    group = RunGroup(model_name=model_name, group_name=group_name)

    return ForcedAdmissionOutpatientPipelineRun(
        group=group,
        name=group.get_best_runs_by_lookahead()[int(model_type), 2],
        pos_rate=0.04,
        create_output_paths_on_init=False,
    )


def stratified_cross_validation(
    cfg: FullConfigSchema,
    pipe: Pipeline,
    train_df: pd.DataFrame,
    train_col_names: list[str],
    outcome_col_name: str,
) -> tuple[pd.DataFrame, list[float]]:
    """Performs stratified and grouped cross validation using the pipeline."""
    msg = Printer(timestamp=True)

    X = train_df[train_col_names]
    y = train_df[outcome_col_name]

    # Create folds
    msg.info("Creating folds")
    msg.info(f"Training on {X.shape[1]} columns and {X.shape[0]} rows")

    folds = StratifiedGroupKFold(n_splits=cfg.n_crossval_splits).split(
        X=X, y=y, groups=train_df[cfg.data.col_name.id]
    )

    # Perform CV and get out of fold predictions
    train_df["oof_y_hat"] = np.nan

    oof_aucs = []

    for i, (train_idxs, val_idxs) in enumerate(folds):
        msg_prefix = f"Fold {i + 1}"

        msg.info(f"{msg_prefix}: Training fold")

        X_train, y_train = (X.loc[train_idxs], y.loc[train_idxs])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X_train)[:, 1]

        msg.info(f"{msg_prefix}: Train AUC = {round(roc_auc_score(y_train,y_pred), 3)}")  # type: ignore

        oof_y_pred = pipe.predict_proba(X.loc[val_idxs])[:, 1]  # type: ignore

        oof_auc = round(roc_auc_score(y.loc[val_idxs], oof_y_pred), 3)  # type: ignore

        msg.info(f"{msg_prefix}: Oof AUC = {oof_auc}")  # type: ignore

        train_df.loc[val_idxs, "oof_y_hat"] = oof_y_pred  # type: ignore

        oof_aucs.append(oof_auc)

    return train_df, oof_aucs


def crossvalidate(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
) -> tuple[pd.DataFrame, list[float]]:
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

    df, oof_aucs = stratified_cross_validation(
        cfg=cfg,
        pipe=pipe,
        train_df=train,
        train_col_names=train_col_names,
        outcome_col_name=outcome_col_name,
    )

    df = df.rename(columns={"oof_y_hat": "y_hat_prob"})

    return create_eval_dataset(
        col_names=cfg.data.col_name, outcome_col_name=outcome_col_name, df=df
    ), oof_aucs  # type: ignore


def load_data(cfg: FullConfigSchema) -> pd.DataFrame:
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

    return train_dataset


def train_model(cfg: FullConfigSchema) -> tuple[float, list[float]]:
    """Train a single model and evaluate it."""
    dataset = load_data(cfg)

    outcome_col_name_for_train, train_col_names = get_col_names(cfg, dataset)

    pipe = create_post_split_pipeline(cfg)

    eval_dataset, oof_aucs = crossvalidate(
        cfg=cfg,
        train=dataset,
        pipe=pipe,
        outcome_col_name=outcome_col_name_for_train,  # type: ignore
        train_col_names=train_col_names,
    )

    roc_auc = roc_auc_score(  # type: ignore
        eval_dataset.y, eval_dataset.y_hat_probs
    )
    return roc_auc, oof_aucs  # type: ignore


def cross_validation_performance_table(
    models_to_train: pd.DataFrame, models_descriptions: str | None = None
) -> pd.DataFrame:
    roc_aucs = []
    cfs = []
    std_devs = []
    oof_intervals = []

    for i in models_to_train.index:
        run = _get_model_pipeline(
            model_name=models_to_train["model_name"][i],  # type: ignore
            group_name=models_to_train["group_name"][i],  # type: ignore
            model_type=models_to_train["model_type"][i],  # type: ignore
        )
        roc_auc, oof_aucs = train_model(run.inputs.cfg)

        high_oof_auc = max(oof_aucs)
        low_oof_auc = min(oof_aucs)
        mean_auroc = np.mean(oof_aucs)  # type: ignore
        std_dev_auroc = np.std(oof_aucs)  # type: ignore
        std_error_auroc = std_dev_auroc / np.sqrt(len(oof_aucs))
        dof = len(oof_aucs) - 1
        conf_interval = stats.t.interval(0.95, dof, loc=mean_auroc, scale=std_error_auroc)

        roc_aucs.append(roc_auc)
        cfs.append(f"[{round(conf_interval[0],3)}:{round(conf_interval[1],3)}]")
        std_devs.append(round(std_dev_auroc, 3))
        oof_intervals.append(f"{low_oof_auc}-{high_oof_auc}")

    df_dict = {
        "Predictor set": models_to_train["pretty_model_name"],
        "Model type": models_to_train["pretty_model_type"],
        "AUROC score": roc_aucs,
        "95 percent confidence interval": cfs,
        "Standard deviation": std_devs,
        "5-fold out-of-fold AUROC interval": oof_intervals,
    }
    df = pd.DataFrame(df_dict)

    EVAL_ROOT = Path("E:/shared_resources/forced_admissions_outpatient/eval")
    df.to_excel(EVAL_ROOT / f"{models_descriptions}cross_validation_table.xlsx", index=False)

    return df


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "pretty_model_name": [
                "Diagnoses",
                "Patient descriptors",
                "Sentence transformer embeddings",
                "TF-IDF features",
                "Full predictor set",
                "Full predictor set 90 days lookahead",
                "Full predictor set 365 days lookahead",
            ],
            "model_name": [
                "limited_model_demographics_diagnoses",
                "full_model_without_text_features",
                "only_sent_trans_model",
                "only_tfidf_750_model_added_konklusion",
                "full_model_with_text_features_added_konklusion",
                "full_model_with_text_features_lookahead_90_added_konklusion",
                "full_model_with_text_features_lookahead_365_added_konklusion",
            ],
            "group_name": [
                "elaborations-piecrust",
                "frustums-liveable",
                "overlogicality-gardenesque",
                "hectical-jawboned",
                "dittoed-tetrastylous",
                "psychorhythmic-unmaneged",
                "alopecias-peewees",
            ],
            # 0 is logistic regression and 1 is Xgboost
            "model_type": [0, 0, 0, 0, 0, 0, 0],
            # change this for either XGboost or Logistic regression
            "pretty_model_type": [
                "Logistic regression",
                "Logistic regression",
                "Logistic regression",
                "Logistic regression",
                "Logistic regression",
                "Logistic regression",
                "Logistic regression",
            ],
        }
    )

    cross_validation_performance_table(df, "primary_models_")
