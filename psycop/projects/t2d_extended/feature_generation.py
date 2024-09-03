"""Main feature generation."""

import datetime
import logging
import multiprocessing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import timeseriesflattener as ts
from tqdm import tqdm

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    t2d_outcome_timestamps,
    t2d_pred_times,
)


def get_t2d_project_info() -> ProjectInfo:
    return ProjectInfo(
        project_name="t2d_extended", project_path=OVARTACI_SHARED_DIR / "t2d_extended"
    )


def _init_t2d_predictor(
    df_loader: Callable[[], pd.DataFrame],
    layer: int,
    fallback: float | int,
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [730]
    ],
    aggregation_fns: Sequence[ts.aggregators.Aggregator] = [
        ts.MeanAggregator()
        # ts.MinAggregator(),
        # ts.MaxAggregator(),
    ],
    column_prefix: str = "pred_layer_{}",
    entity_id_col_name: str = "dw_ek_borger",
) -> ts.PredictorSpec:
    logging.info(f"Initialising {df_loader.__name__}")
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


@dataclass(frozen=True)
class LayerSpecPair:
    layer: int
    spec: AnySpec | BooleanSpec | ContinuousSpec | CategoricalSpec


def _pair_to_spec(pair: LayerSpecPair) -> AnySpec:
    match pair.spec:
        case ContinuousSpec():
            return _init_t2d_predictor(
                df_loader=pair.spec.loader, layer=pair.layer, fallback=np.NaN
            )
        case BooleanSpec():
            return _init_t2d_predictor(df_loader=pair.spec.loader, layer=pair.layer, fallback=0.0)
        case CategoricalSpec():
            return _init_t2d_predictor(
                df_loader=pair.spec.loader, layer=pair.layer, fallback=np.NaN
            )
        case ts.PredictorSpec() | ts.OutcomeSpec() | ts.StaticSpec() | ts.TimeDeltaSpec():
            return pair.spec


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
                    init_df=t2d_outcome_timestamps().frame, entity_id_col_name="dw_ek_borger"
                ),
                lookahead_distances=[datetime.timedelta(days=365 * 5)],
                aggregators=[ts.MaxAggregator()],
                fallback=0,
                column_prefix="outc_t2d",
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
        1: [ContinuousSpec(hba1c)],
    }

    layer_spec_pairs = [
        LayerSpecPair(layer, spec)
        for layer, spec_list in feature_layers.items()
        for spec in spec_list
    ]

    # Run in parallel for faster loading
    logging.info("Loading specifications")
    with multiprocessing.Pool(processes=15) as pool:
        specs = list(tqdm(pool.imap(_pair_to_spec, layer_spec_pairs), total=len(layer_spec_pairs)))

    logging.info("Generating feature set")
    generate_feature_set(
        project_info=get_t2d_project_info(),
        eligible_prediction_times_frame=t2d_pred_times().prediction_times,
        feature_specs=specs,
        feature_set_name="t2d_extended_feature_set",
        n_workers=10,
        step_size=datetime.timedelta(days=365),
        do_dataset_description=False,
    )
