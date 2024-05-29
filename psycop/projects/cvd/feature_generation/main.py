"""Main feature generation."""

import datetime
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
import timeseriesflattener as ts
from timeseriesflattener.v1.aggregation_fns import mean

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
    generate_feature_set_tsflattener_v1,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    chronic_lung_disease,
    f0_disorders,
    f1_disorders,
    f2_disorders,
    f3_disorders,
    f4_disorders,
    f5_disorders,
    f6_disorders,
    f7_disorders,
    f8_disorders,
    f9_disorders,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c, hdl, ldl
from psycop.common.feature_generation.loaders.raw.load_medications import (
    top_10_weight_gaining_antipsychotics,
)
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    smoking_categorical,
    smoking_continuous,
    systolic_blood_pressure,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    CVDCohortDefiner,
    cvd_outcome_timestamps,
    cvd_pred_times,
)
from psycop.projects.cvd.feature_generation.specify_features import CVDFeatureSpecifier


def get_cvd_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="cvd", project_path=OVARTACI_SHARED_DIR / "cvd" / "feature_set")


def cvd_pred(
    init_df: pl.LazyFrame | pl.DataFrame | pd.DataFrame,
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [90, 365, 730]
    ],
    aggregation_fns: Sequence[ts.aggregators.Aggregator] = [
        ts.MeanAggregator(),
        ts.MinAggregator(),
        ts.MaxAggregator(),
    ],
    column_prefix: str = "pred",
) -> ts.PredictorSpec:
    return ts.PredictorSpec(
        value_frame=ts.ValueFrame(init_df=init_df),
        lookbehind_distances=lookbehind_distances,
        aggregators=aggregation_fns,
        fallback=np.nan,
        column_prefix=column_prefix,
    )


if __name__ == "__main__":
    project_info = get_cvd_project_info()
    eligible_prediction_times = cvd_pred_times()

    layer = 1
    feature_layers: Mapping[
        int, Sequence[ts.OutcomeSpec | ts.StaticSpec | ts.TimeDeltaSpec | ts.PredictorSpec]
    ] = {
        1: [
            ts.OutcomeSpec(
                value_frame=ts.ValueFrame(init_df=cvd_outcome_timestamps().frame),
                lookahead_distances=[datetime.timedelta(days=365 * 5)],
                aggregators=[ts.MaxAggregator()],
                fallback=0,
            ),
            ts.StaticSpec(
                value_frame=ts.StaticFrame(init_df=sex_female()),
                fallback=0,
                column_prefix="pred_layer_1",
            ),
            ts.TimeDeltaSpec(
                init_frame=ts.TimestampValueFrame(
                    birthdays(),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="date_of_birth",
                ),
                fallback=0,
                output_name="pred_layer_1_age",
            ),
            cvd_pred(ldl(), column_prefix="pred_layer_1"),
            cvd_pred(systolic_blood_pressure(), column_prefix="pred_layer_1"),
        ],
        2: [
            cvd_pred(smoking_continuous(), column_prefix="pred_layer_2"),
            cvd_pred(smoking_categorical(), column_prefix="pred_layer_2"),
        ],
        3: [
            cvd_pred(hba1c(), column_prefix="pred_layer_3"),
            cvd_pred(chronic_lung_disease(), column_prefix="pred_layer_3"),
        ],
        4: [
            cvd_pred(f0_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(f1_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(f2_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(f3_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(f4_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(f5_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(f6_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(f7_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(f8_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(f9_disorders(), column_prefix="pred_layer_4"),
            cvd_pred(top_10_weight_gaining_antipsychotics(), column_prefix="pred_layer_4"),
            cvd_pred(hdl(), column_prefix="pred_layer_4"),
        ],
    }

    generate_feature_set(
        project_info=project_info,
        eligible_prediction_times_frame=eligible_prediction_times,
        feature_specs=[spec for layer_specs in feature_layers.values() for spec in layer_specs],
        feature_set_name="cvd_feature_set",
        n_workers=5,
        step_size=datetime.timedelta(days=365),
        do_dataset_description=False,
    )
