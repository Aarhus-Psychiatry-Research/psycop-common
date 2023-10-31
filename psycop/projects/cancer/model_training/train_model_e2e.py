import logging

import pandas as pd

from psycop.common.model_training.application_modules.train_model.main import (
    train_model,
)
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
from psycop.projects.cancer.feature_generation.main import get_cancer_project_info

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


if __name__ == "__main__":
    project_info = get_cancer_project_info()

    cfg = FullConfigSchema(
        project=ProjectSchema(
            wandb=WandbSchema(
                group="days",
                mode="offline",
                entity="frihae",
            ),
            project_path=project_info.project_path,
            name=project_info.project_name,
            seed=42,
            gpu=True,
        ),
        data=DataSchema(
            dir=project_info.flattened_dataset_dir / "breast_cancer_feature_set",
            splits_for_training=["train"],
            col_name=ColumnNamesSchema(
                age="pred_age_in_years",
                is_female="pred_sex_female",
                id="dw_ek_borger",
                outcome_timestamp=None,
            ),
            pred_prefix=project_info.prefix.predictor,
            outc_prefix=project_info.prefix.outcome,
            n_training_samples=None,
        ),
        preprocessing=PreprocessingConfigSchema(
            pre_split=PreSplitPreprocessingConfigSchema(
                min_lookahead_days=1825,
                min_age=18,
                min_prediction_time_date="2013-01-01",
                lookbehind_combination=[730],
                drop_visits_after_exclusion_timestamp=False,
                drop_rows_after_outcome=False,
            ),
            post_split=PostSplitPreprocessingConfigSchema(
                imputation_method="mean",
                scaling=None,
                feature_selection=FeatureSelectionSchema(),
            ),
        ),
        model=ModelConfSchema(
            name="xgboost",
            require_imputation=True,
            args={"max_depth": 2},
        ),
        n_crossval_splits=10,
    )

    auroc = train_model(cfg=cfg)
    print(auroc)

