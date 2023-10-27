"""Experiments with models trained and evaluated using different lookaheads for the outcome columns"""


import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.common.model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop.common.model_training.training.utils import create_eval_dataset
from psycop.common.model_training.training_output.dataclasses import EvalDataset
from psycop.common.model_training.utils.col_name_inference import get_col_names
from psycop.projects.forced_admission_inpatient.utils.pipeline_objects import (
    ForcedAdmissionInpatientPipelineRun,
)


def train_validate(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name_for_train: str,
    outcome_col_name_for_test: str,
    train_col_names: list[str],
) -> EvalDataset:
    """Train model on pre-defined train and validation split and return
    evaluation dataset.

    Args:
        cfg (FullConfig): Config object
        train: Training dataset
        val: Validation dataset
        pipe: Pipeline
        outcome_col_name_for_train: Name of the outcome column for model training
        outcome_col_name_for_test: Name of the outcome column for model evaluation
        train_col_names: Names of the columns to use for training

    Returns:
        Evaluation dataset
    """

    X_train = train[train_col_names]
    y_train = train[outcome_col_name_for_train]
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
        outcome_col_name=outcome_col_name_for_test,
        df=df,
    )


def load_data(
    cfg: FullConfigSchema,
) -> list[pd.DataFrame]:
    """Train a single model and evaluate it."""
    train_dataset = pd.concat(
        [
            load_and_filter_split_from_cfg(
                data_cfg=cfg.data,
                pre_split_cfg=cfg.preprocessing.pre_split,
                split=split,
            )
            for split in cfg.data.splits_for_training
        ],
        ignore_index=True,
    )

    eval_dataset = pd.concat(
        [
            load_and_filter_split_from_cfg(
                data_cfg=cfg.data,
                pre_split_cfg=cfg.preprocessing.pre_split,
                split=split,  # type: ignore
            )
            for split in cfg.data.splits_for_evaluation  # type: ignore
        ],
        ignore_index=True,
    )

    return [train_dataset, eval_dataset]


def train_model(
    cfg: FullConfigSchema,
    outcome_col_name_for_test: str,
) -> float:
    """Train a single model and evaluate it."""
    datasets = load_data(cfg)

    outcome_col_name_for_train, train_col_names = get_col_names(cfg, datasets[0])

    pipe = create_post_split_pipeline(cfg)

    eval_dataset = train_validate(
        cfg=cfg,
        train=datasets[0],
        val=datasets[1],
        pipe=pipe,
        outcome_col_name_for_train=outcome_col_name_for_train,  # type: ignore
        outcome_col_name_for_test=outcome_col_name_for_test,  # type: ignore
        train_col_names=train_col_names,
    )

    roc_auc = roc_auc_score(  # type: ignore
        eval_dataset.y,
        eval_dataset.y_hat_probs,
    )
    return roc_auc  # type: ignore


def main(run: ForcedAdmissionInpatientPipelineRun):
    cfg = run.inputs.cfg

    train_model(
        cfg=cfg,
        outcome_col_name_for_test="outc_forced_admissions_within_180_days_maximum_fallback_0_dichotomous",
    )


if __name__ == "__main__":
    from psycop.projects.forced_admission_inpatient.model_eval.selected_runs import (
        get_best_eval_pipeline,
    )

    main(run=get_best_eval_pipeline())
