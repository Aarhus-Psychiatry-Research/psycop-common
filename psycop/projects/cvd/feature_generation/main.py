"""Main feature generation."""

import datetime
import functools
import logging
import multiprocessing
import multiprocessing.pool
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import timeseriesflattener as ts
from timeseriesflattener.v1.aggregation_fns import mean
from tqdm import tqdm

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    atrial_fibrillation,
    chronic_kidney_failure,
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
    pectoral_angina,
    type_1_diabetes,
    type_2_diabetes,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import (
    hba1c,
    hdl,
    ldl,
    total_cholesterol,
)
from psycop.common.feature_generation.loaders.raw.load_medications import (
    antihypertensives,
    top_10_weight_gaining_antipsychotics,
)
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    bmi,
    height_in_cm,
    smoking_categorical,
    smoking_continuous,
    systolic_blood_pressure,
    weight_in_kg,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    cvd_outcome_timestamps,
    cvd_pred_times,
)


def get_cvd_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="cvd", project_path=OVARTACI_SHARED_DIR / "cvd" / "feature_set")


def _init_cvd_predictor(
    df_loader: Callable[[], pd.DataFrame],
    layer: int,
    fallback: float | int,
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [90, 365, 730]
    ],
    aggregation_fns: Sequence[ts.aggregators.Aggregator] = [
        ts.MeanAggregator(),
        ts.MinAggregator(),
        ts.MaxAggregator(),
    ],
    column_prefix: str = "pred_layer_{}",
    entity_id_col_name: str = "dw_ek_borger",
) -> ts.PredictorSpec:
    return ts.PredictorSpec(
        value_frame=ts.ValueFrame(
            init_df=pl.from_pandas(df_loader()).rename({"value": df_loader.__name__}),
            entity_id_col_name=entity_id_col_name,
        ),
        lookbehind_distances=lookbehind_distances,
        aggregators=aggregation_fns,
        fallback=fallback,
        column_prefix=column_prefix.format(layer),
    )


AnySpec = ts.PredictorSpec | ts.OutcomeSpec | ts.StaticSpec | ts.TimeDeltaSpec


@dataclass(frozen=True)
class ContinuousSpec:
    loader: Callable[[], pd.DataFrame]


@dataclass(frozen=True)
class BooleanSpec:
    loader: Callable[[], pd.DataFrame]


@dataclass(frozen=True)
class CategoricalSpec:
    loader: Callable[[], pd.DataFrame]


def _pair_to_spec(
    pair: tuple[int, AnySpec | BooleanSpec | ContinuousSpec | CategoricalSpec],
) -> AnySpec:
    layer = pair[0]
    spec = pair[1]
    match spec:
        case ContinuousSpec():
            return _init_cvd_predictor(df_loader=spec.loader, layer=layer, fallback=np.NaN)
        case BooleanSpec():
            return _init_cvd_predictor(df_loader=spec.loader, layer=layer, fallback=0.0)
        case CategoricalSpec():
            return _init_cvd_predictor(df_loader=spec.loader, layer=layer, fallback=np.NaN)
        case ts.PredictorSpec() | ts.OutcomeSpec() | ts.StaticSpec() | ts.TimeDeltaSpec():
            return spec


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

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
        1: [ContinuousSpec(ldl), ContinuousSpec(systolic_blood_pressure)],
        2: [ContinuousSpec(smoking_continuous), CategoricalSpec(smoking_categorical)],
        3: [ContinuousSpec(hba1c), ContinuousSpec(chronic_lung_disease)],
        4: [
            BooleanSpec(f0_disorders),
            BooleanSpec(f1_disorders),
            BooleanSpec(f2_disorders),
            BooleanSpec(f3_disorders),
            BooleanSpec(f4_disorders),
            BooleanSpec(f5_disorders),
            BooleanSpec(f6_disorders),
            BooleanSpec(f7_disorders),
            BooleanSpec(f8_disorders),
            BooleanSpec(f9_disorders),
            BooleanSpec(top_10_weight_gaining_antipsychotics),
            ContinuousSpec(hdl),
        ],
        5: [BooleanSpec(atrial_fibrillation), BooleanSpec(antihypertensives)],
        6: [
            BooleanSpec(type_1_diabetes),
            BooleanSpec(type_2_diabetes),
            ContinuousSpec(weight_in_kg),
            ContinuousSpec(height_in_cm),
            ContinuousSpec(bmi),
        ],
        7: [
            ContinuousSpec(total_cholesterol),
            BooleanSpec(chronic_kidney_failure),
            BooleanSpec(pectoral_angina),
        ],
    }

    layer_spec_pairs = [
        (layer, spec) for layer, spec_list in feature_layers.items() for spec in spec_list
    ]

    # Run in parallel for faster loading
    logging.info("Loading specifications")
    with multiprocessing.Pool(processes=15) as pool:
        specs = list(tqdm(pool.imap(_pair_to_spec, layer_spec_pairs), total=len(layer_spec_pairs)))

    logging.info("Generating feature set")
    generate_feature_set(
        project_info=get_cvd_project_info(),
        eligible_prediction_times_frame=cvd_pred_times(),
        feature_specs=specs,
        feature_set_name="cvd_feature_set",
        n_workers=10,
        step_size=datetime.timedelta(days=365),
        do_dataset_description=False,
    )
