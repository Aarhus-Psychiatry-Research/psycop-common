from pathlib import Path

import pandas as pd
from timeseriesflattener.aggregation_fns import maximum, minimum
from timeseriesflattener.feature_specs.single_specs import (
    OutcomeSpec,
    PredictorSpec,
)

from psycop.common.feature_generation.application_modules.project_setup import (
    ColNames,
    Prefixes,
    ProjectInfo,
)
from psycop.common.minimal_pipeline import minimal_pipeline
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
from psycop.common.test_utils.str_to_df import str_to_df


def test_e2e(tmp_path: Path):
    """Test e2e."""
    project_info = ProjectInfo(
        project_name="test_project",
        project_path=tmp_path,
        prefix=Prefixes(),
        col_names=ColNames(id="entity_id"),  # type: ignore
    )

    prediction_times_df = str_to_df(
        """entity_id,timestamp
999,2000-01-02, # Late start of dataset to avoid filtering
1,2020-01-02 00:00:00,
2,2020-01-02 00:00:00,
3,2020-01-02 00:00:00,
4,2020-01-02 00:00:00,
5,2020-01-02 00:00:00,
6,2020-01-02 00:00:00,
999,2030-01-01, # Late end of dataset to avoid filtering
""",
    )

    prediction_times_dfs = []

    for i in range(100):
        augmented_df = prediction_times_df.copy()
        augmented_df["timestamp"] = augmented_df["timestamp"] + pd.Timedelta(seconds=i)
        prediction_times_dfs.append(augmented_df)

    predictor_df = str_to_df(
        """entity_id,timestamp,value
1,2020-01-01,1
2,2020-01-01,0
3,2020-01-01,1
4,2020-01-01,0
5,2020-01-01,1
6,2020-01-01,0
""",
    )

    outcome_df = str_to_df(
        """entity_id,timestamp,value
1,2020-01-03,1
2,2020-01-03,0
3,2020-01-03,1
4,2020-01-03,0
5,2020-01-03,1
6,2020-01-03,0
""",
    )

    feature_specs = [
        PredictorSpec(
            feature_base_name="test_predictor",
            aggregation_fn=minimum,
            lookbehind_days=3,
            timeseries_df=predictor_df,
            fallback=0,
        ),
        OutcomeSpec(
            feature_base_name="test_outcome",
            aggregation_fn=maximum,
            lookahead_days=3,
            incident=False,
            timeseries_df=outcome_df,
            fallback=0,
        ),
    ]

    prediction_times_df = pd.concat(prediction_times_dfs)

    train_ids_df = prediction_times_df[["entity_id"]].drop_duplicates()

    model_training_cfg = FullConfigSchema(
        project=ProjectSchema(
            wandb=WandbSchema(
                group="test_group",
                mode="offline",
                entity="test_entity",
            ),
            project_path=project_info.project_path,
            name=project_info.project_name,
            seed=42,
            gpu=False,
        ),
        data=DataSchema(
            dir=project_info.flattened_dataset_dir,
            splits_for_training=["train"],
            col_name=ColumnNamesSchema(
                age=None,
                is_female=None,
                id="entity_id",
                outcome_timestamp=None,
            ),
            pred_prefix=project_info.prefix.predictor,
            outc_prefix=project_info.prefix.outcome,
            n_training_samples=None,
        ),
        preprocessing=PreprocessingConfigSchema(
            pre_split=PreSplitPreprocessingConfigSchema(
                min_lookahead_days=0,
                min_age=0,
                min_prediction_time_date="2020-01-01",
                lookbehind_combination=[3],
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
        n_crossval_splits=2,
    )

    auroc = minimal_pipeline(
        project_info=project_info,
        feature_specs=feature_specs,
        prediction_times_df=prediction_times_df,
        add_birthdays=False,
        model_training_cfg=model_training_cfg,
        split2ids_df={"train": train_ids_df},
        split_names=["train"],
    )

    assert auroc == 1.0


def test_e2e_multilabel(tmp_path: Path):
    """Test e2e."""
    project_info = ProjectInfo(
        project_name="test_project",
        project_path=tmp_path,
        prefix=Prefixes(),
        col_names=ColNames(id="entity_id"),  # type: ignore
    )

    prediction_times_df = str_to_df(
        """entity_id,timestamp
999,2000-01-02, # Late start of dataset to avoid filtering
1,2020-01-02 00:00:00,
2,2020-01-02 00:00:00,
3,2020-01-02 00:00:00,
4,2020-01-02 00:00:00,
5,2020-01-02 00:00:00,
6,2020-01-02 00:00:00,
999,2030-01-01, # Late end of dataset to avoid filtering
""",
    )

    prediction_times_dfs = []

    for i in range(100):
        augmented_df = prediction_times_df.copy()
        augmented_df["timestamp"] = augmented_df["timestamp"] + pd.Timedelta(seconds=i)
        prediction_times_dfs.append(augmented_df)

    predictor_df = str_to_df(
        """entity_id,timestamp,value
1,2020-01-01,1
2,2020-01-01,0
3,2020-01-01,1
4,2020-01-01,0
5,2020-01-01,1
6,2020-01-01,0
""",
    )

    outcome1_df = str_to_df(
        """entity_id,timestamp,value
1,2020-01-03,1
2,2020-01-03,0
3,2020-01-03,1
4,2020-01-03,0
5,2020-01-03,1
6,2020-01-03,0
""",
    )

    outcome2_df = str_to_df(
        """entity_id,timestamp,value
1,2020-01-03,1
2,2020-01-03,0
3,2020-01-03,1
4,2020-01-03,0
5,2020-01-03,1
6,2020-01-03,0
""",
    )

    feature_specs = [
        PredictorSpec(
            feature_base_name="test_predictor",
            aggregation_fn=minimum,
            lookbehind_days=3,
            timeseries_df=predictor_df,
            fallback=0,
        ),
        OutcomeSpec(
            feature_base_name="test_outcome1",
            aggregation_fn=maximum,
            lookahead_days=3,
            incident=False,
            timeseries_df=outcome1_df,
            fallback=0,
        ),
        OutcomeSpec(
            feature_base_name="test_outcome2",
            aggregation_fn=maximum,
            lookahead_days=3,
            incident=False,
            timeseries_df=outcome2_df,
            fallback=0,
        ),
    ]

    prediction_times_df = pd.concat(prediction_times_dfs)

    train_ids_df = prediction_times_df[["entity_id"]].drop_duplicates()

    model_training_cfg = FullConfigSchema(
        project=ProjectSchema(
            wandb=WandbSchema(
                group="test_group",
                mode="offline",
                entity="test_entity",
            ),
            project_path=project_info.project_path,
            name=project_info.project_name,
            seed=42,
            gpu=False,
        ),
        data=DataSchema(
            dir=project_info.flattened_dataset_dir,
            splits_for_training=["train"],
            col_name=ColumnNamesSchema(
                age=None,
                is_female=None,
                id="entity_id",
                outcome_timestamp=None,
            ),
            pred_prefix=project_info.prefix.predictor,
            outc_prefix=project_info.prefix.outcome,
            n_training_samples=None,
        ),
        preprocessing=PreprocessingConfigSchema(
            pre_split=PreSplitPreprocessingConfigSchema(
                min_lookahead_days=0,
                min_age=0,
                min_prediction_time_date="2020-01-01",
                lookbehind_combination=[3],
                drop_visits_after_exclusion_timestamp=False,
                drop_rows_after_outcome=False,
                keep_only_one_outcome_col=False,
                classification_objective="multilabel",
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
        n_crossval_splits=2,
    )

    auroc = minimal_pipeline(
        project_info=project_info,
        feature_specs=feature_specs,
        prediction_times_df=prediction_times_df,
        add_birthdays=False,
        model_training_cfg=model_training_cfg,
        split2ids_df={"train": train_ids_df},
        split_names=["train"],
    )

    assert auroc == 1
