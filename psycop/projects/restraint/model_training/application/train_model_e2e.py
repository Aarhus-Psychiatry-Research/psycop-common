import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from psycop.common.feature_generation.application_modules.project_setup import (
    Prefixes,
    ProjectInfo,
)
from psycop.common.model_training.application_modules.train_model.main import (
    get_eval_dir,
)
from psycop.common.model_training.application_modules.wandb_handler import WandbHandler
from psycop.common.model_training.config_schemas.data import (
    ColumnNamesSchema,
    DataSchema,
)
from psycop.common.model_training.config_schemas.full_config import FullConfigSchema
from psycop.common.model_training.config_schemas.model import ModelConfSchema
from psycop.common.model_training.config_schemas.preprocessing import (
    FeatureSelectionSchema,
    PostSplitPreprocessingConfigSchema,
    PreprocessingConfigSchema,
    PreSplitPreprocessingConfigSchema,
)
from psycop.common.model_training.config_schemas.project import (
    ProjectSchema,
    WandbSchema,
)
from psycop.common.model_training.data_loader.utils import (
    load_and_filter_split_from_cfg,
)
from psycop.common.model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop.common.model_training.training.train_and_predict import train_and_predict
from psycop.common.model_training.training_output.model_evaluator import ModelEvaluator
from psycop.common.model_training.utils.col_name_inference import get_col_names

log = logging.getLogger(__name__)


def load_datasets(cfg: FullConfigSchema) -> pd.DataFrame:
    train_datasets = pd.concat(
        [
            load_and_filter_split_from_cfg(
                data_cfg=cfg.data,
                pre_split_cfg=cfg.preprocessing.pre_split,
                split=cfg.data.splits_for_training[0],
            ),
        ],
        ignore_index=True,
    )

    return train_datasets


def train_remove_days(cfg: FullConfigSchema, train_datasets: pd.DataFrame):
    WandbHandler(cfg=cfg).setup_wandb()

    pipe = create_post_split_pipeline(cfg)
    outcome_col_name, train_col_names = get_col_names(cfg, train_datasets)

    for day in range(160, 16, -10):
        train_dataset = train_datasets[
            train_datasets.pred_adm_day_count < day
        ].reset_index(drop=True)

        eval_dataset = train_and_predict(
            cfg=cfg,
            train_datasets=train_dataset,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
        )

        eval_dir = get_eval_dir(cfg)

        auroc = ModelEvaluator(
            eval_dir_path=eval_dir,
            cfg=cfg,
            pipe=pipe,
            eval_ds=eval_dataset,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
        ).evaluate_and_save_eval_data()

        with Path.open("outputs/log.log", "a+") as log:  # type: ignore
            log.write(f"{day}: {auroc}\n")


def train_add_features(cfg: FullConfigSchema, train_datasets: pd.DataFrame):
    WandbHandler(cfg=cfg).setup_wandb()
    pipe = create_post_split_pipeline(cfg)
    outcome_col_name, train_col_names = get_col_names(cfg, train_datasets)

    pred_types = np.unique(
        [re.findall(r"pred_(.+)?_within", x) for x in train_col_names],
    )
    pred_types = [x for x in pred_types if x != []]
    predictor_names = [
        "pred_adm_day_count",
        "pred_age_in_years",
        "pred_sex_female",
        "pred_broeset_violence_checklist_within_1_days_mean_fallback_nan",
    ]

    eval_dataset = train_and_predict(
        cfg=cfg,
        train_datasets=train_datasets,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=predictor_names,
    )

    eval_dir = get_eval_dir(cfg)

    auroc = ModelEvaluator(
        eval_dir_path=eval_dir,
        cfg=cfg,
        pipe=pipe,
        eval_ds=eval_dataset,
        outcome_col_name=outcome_col_name,
        train_col_names=predictor_names,
    ).evaluate_and_save_eval_data()

    with Path.open("outputs/log.log", "a+") as log:  # type: ignore
        log.write(f"Baseline: {auroc}\n")

    for pred_type in pred_types:
        predictor_names += [
            c for c in train_col_names if pred_type[0] in c and c not in predictor_names
        ]

        eval_dataset = train_and_predict(
            cfg=cfg,
            train_datasets=train_datasets,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=predictor_names,
        )

        eval_dir = get_eval_dir(cfg)

        auroc = ModelEvaluator(
            eval_dir_path=eval_dir,
            cfg=cfg,
            pipe=pipe,
            eval_ds=eval_dataset,
            outcome_col_name=outcome_col_name,
            train_col_names=predictor_names,
        ).evaluate_and_save_eval_data()

        with Path.open("outputs/log.log", "a+") as log:  # type: ignore
            log.write(f"{pred_type}: {auroc}\n")


def train_multilabel(cfg: FullConfigSchema, train_datasets: pd.DataFrame):
    WandbHandler(cfg=cfg).setup_wandb()

    pipe = create_post_split_pipeline(cfg)
    outcome_col_name, train_col_names = get_col_names(cfg, train_datasets)

    train_datasets = train_datasets.drop(
        columns=[
            "outcome_coercion_bool_within_2_days",
            "outcome_coercion_type_within_2_days",
        ],
    )

    eval_dataset = train_and_predict(
        cfg=cfg,
        train_datasets=train_datasets,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
    )

    eval_dir = get_eval_dir(cfg)

    auroc = ModelEvaluator(
        eval_dir_path=eval_dir,
        cfg=cfg,
        pipe=pipe,
        eval_ds=eval_dataset,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
    ).evaluate_and_save_eval_data()

    print(auroc)


if __name__ == "__main__":
    project_info = ProjectInfo(
        project_name="e2e",
        project_path=Path("E:/shared_resources/coercion/e2e"),
        prefix=Prefixes(outcome="outcome_coercion_bool"),
    )

    cfg = FullConfigSchema(
        project=ProjectSchema(
            wandb=WandbSchema(
                group="days",
                mode="offline",
                entity="test_entity",
            ),
            project_path=project_info.project_path,
            name=project_info.project_name,
            seed=42,
            gpu=True,
        ),
        data=DataSchema(
            dir=project_info.flattened_dataset_dir,
            splits_for_training=["train"],
            col_name=ColumnNamesSchema(
                age="pred_age_in_years",
                is_female="pred_sex_female",
                id="dw_ek_borger",
                outcome_timestamp="timestamp_outcome",
            ),
            pred_prefix=project_info.prefix.predictor,
            outc_prefix=project_info.prefix.outcome,
            n_training_samples=None,
        ),
        preprocessing=PreprocessingConfigSchema(
            pre_split=PreSplitPreprocessingConfigSchema(
                min_lookahead_days=2,
                min_age=0,
                min_prediction_time_date="2015-01-01",
                lookbehind_combination=[1, 3, 7, 10, 30, 180, 365, 730],
                drop_visits_after_exclusion_timestamp=False,
                drop_rows_after_outcome=False,
            ),
            post_split=PostSplitPreprocessingConfigSchema(
                imputation_method=None,
                scaling=None,
                feature_selection=FeatureSelectionSchema(),
            ),
        ),
        model=ModelConfSchema(
            name="xgboost",
            require_imputation=False,
            args={},
        ),
        n_crossval_splits=3,
    )

    data = load_datasets(cfg)
    train_remove_days(cfg, data)
