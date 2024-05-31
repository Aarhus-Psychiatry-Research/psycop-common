"""Main feature generation."""

import datetime
import functools
import logging
from collections.abc import Mapping, Sequence
from multiprocessing import Pool
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import timeseriesflattener as ts
from timeseriesflattener.v1.aggregation_fns import mean

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
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


def get_cvd_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="cvd", project_path=OVARTACI_SHARED_DIR / "cvd" / "feature_set")


def cvd_pred(
    init_df: Callable[[], pd.DataFrame],
    layer: int,
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [90, 365, 730]
    ],
    aggregation_fns: Sequence[ts.aggregators.Aggregator] = [
        ts.MeanAggregator(),
        ts.MinAggregator(),
        ts.MaxAggregator(),
    ],
) -> ts.PredictorSpec:
    return ts.PredictorSpec(
        value_frame=ts.ValueFrame(
            init_df=pl.from_pandas(init_df()).rename({"value": init_df.__name__}),
            entity_id_col_name="dw_ek_borger",
        ),
        lookbehind_distances=lookbehind_distances,
        aggregators=aggregation_fns,
        fallback=np.nan,
        column_prefix=f"pred_layer_{layer}",
    )


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    project_info = get_cvd_project_info()
    eligible_prediction_times = cvd_pred_times()

    layer = 1

    feature_layers = {
        0: [
            ts.OutcomeSpec(
                value_frame=ts.ValueFrame(
                    init_df=cvd_outcome_timestamps().frame, entity_id_col_name="dw_ek_borger"
                ),
                lookahead_distances=[datetime.timedelta(days=365 * 5)],
                aggregators=[ts.MaxAggregator()],
                fallback=0,
                column_prefix="outc_cvd",
            ),
            ts.StaticSpec(
                value_frame=ts.StaticFrame(init_df=sex_female(), entity_id_col_name="dw_ek_borger"),
                fallback=0,
                column_prefix="pred",
            ),
            ts.TimeDeltaSpec(
                init_frame=ts.TimestampValueFrame(
                    birthdays(),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="date_of_birth",
                ),
                column_prefix="pred",
                fallback=0,
                output_name="age",
            ),
        ],
        1: [ldl, systolic_blood_pressure],
        2: [smoking_continuous, smoking_categorical],
        3: [hba1c, chronic_lung_disease],
        4: [
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
            top_10_weight_gaining_antipsychotics,
            hdl,
        ],
    }

    feature_specs = []
    for layer, feature in feature_layers.items():
        for spec in feature:
            if isinstance(spec, Callable):
                feature_specs.append(cvd_pred(init_df=spec, layer=layer))
            else:
                feature_specs.append(spec)

    generate_feature_set(
        project_info=project_info,
        eligible_prediction_times_frame=eligible_prediction_times,
        feature_specs=feature_specs,
        feature_set_name="cvd_feature_set",
        n_workers=10,
        step_size=datetime.timedelta(days=365),
        do_dataset_description=False,
    )
