"""Main feature generation."""

import datetime
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import timeseriesflattener as ts
from timeseriesflattener.aggregators import HasValuesAggregator, MeanAggregator

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR, TEXT_EMBEDDINGS_DIR
from psycop.projects.clozapine.feature_generation.cohort_definition.clozapine_cohort_definition import (
    clozapine_pred_times,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.combine_text_structured_clozapine_outcome import (
    combine_structured_and_text_outcome,
)
from psycop.projects.clozapine.loaders.demographics import birthdays, sex_female

TEXT_FILE_NAME = "not_labelled_yet_clozapine_text.parquet"


def get_clozapine_project_info() -> ProjectInfo:
    return ProjectInfo(
        project_name="clozapine",
        project_path=OVARTACI_SHARED_DIR / "clozapine" / "flattened_datasets",
    )


def _init_clozapine_predictor(
    df_loader: Callable[[], pd.DataFrame],
    layer: str,
    fallback: float | int,
    aggregation_fns: Sequence[ts.aggregators.Aggregator],
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [30, 180, 365]
    ],
    column_prefix: str = "pred_layer_{}",
    entity_id_col_name: str = "dw_ek_borger",
    name_overwrite: str | None = None,
) -> ts.PredictorSpec:
    logging.info(f"Initialising {df_loader.__name__ if name_overwrite is None else name_overwrite}")
    return ts.PredictorSpec(
        value_frame=ts.ValueFrame(
            init_df=pl.from_pandas(df_loader()).rename(
                {"value": df_loader.__name__ if name_overwrite is None else name_overwrite}
            ),
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
    aggregation_fns: Sequence[ts.aggregators.Aggregator]
    fallback: float
    name_overwrite: str | None = None


@dataclass(frozen=True)
class BooleanSpec:
    loader: Callable[[], pd.DataFrame]
    aggregation_fns: Sequence[ts.aggregators.Aggregator] = (HasValuesAggregator(),)
    fallback: float = 0.0
    name_overwrite: str | None = None


@dataclass(frozen=True)
class CategoricalSpec:
    loader: Callable[[], pd.DataFrame]
    aggregation_fns: Sequence[ts.aggregators.Aggregator]
    fallback: float
    name_overwrite: str | None = None


@dataclass(frozen=True)
class LayerSpecPair:
    layer: str
    spec: AnySpec | BooleanSpec | ContinuousSpec | CategoricalSpec


def _pair_to_spec(pair: LayerSpecPair) -> AnySpec:
    match pair.spec:
        case ContinuousSpec():
            return _init_clozapine_predictor(
                df_loader=pair.spec.loader,
                layer=pair.layer,
                fallback=pair.spec.fallback,
                aggregation_fns=pair.spec.aggregation_fns,
                name_overwrite=pair.spec.name_overwrite,
            )
        case BooleanSpec():
            return _init_clozapine_predictor(
                df_loader=pair.spec.loader,
                layer=pair.layer,
                fallback=pair.spec.fallback,
                aggregation_fns=pair.spec.aggregation_fns,
                name_overwrite=pair.spec.name_overwrite,
            )
        case CategoricalSpec():
            return _init_clozapine_predictor(
                df_loader=pair.spec.loader,
                layer=pair.layer,
                fallback=pair.spec.fallback,
                aggregation_fns=pair.spec.aggregation_fns,
                name_overwrite=pair.spec.name_overwrite,
            )
        case ts.PredictorSpec() | ts.OutcomeSpec() | ts.StaticSpec() | ts.TimeDeltaSpec():
            return pair.spec


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        stream=sys.stdout,
    )

    pred_times = clozapine_pred_times()

    feature_layers = {
        "basic": [
            ts.OutcomeSpec(
                value_frame=ts.ValueFrame(
                    init_df=combine_structured_and_text_outcome(),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookahead_distances=[datetime.timedelta(days=365)],
                aggregators=[ts.MaxAggregator()],
                fallback=0,
                column_prefix="outc_bp",
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
        "layer_text": [
            ts.PredictorSpec(
                value_frame=ts.ValueFrame(
                    init_df=pl.read_parquet(TEXT_EMBEDDINGS_DIR / TEXT_FILE_NAME).drop(
                        "overskrift"
                    ),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookbehind_distances=[datetime.timedelta(days=180)],
                aggregators=[MeanAggregator()],
                fallback=np.nan,
                column_prefix="pred_layer_text",
            )
        ],
    }

    layer_spec_pairs = [
        LayerSpecPair(layer, spec)
        for layer, spec_list in feature_layers.items()
        for spec in spec_list
    ]

    logging.info("Loading specifications")
    specs = [_pair_to_spec(layer_spec) for layer_spec in layer_spec_pairs]

    logging.info("Generating feature set")
    generate_feature_set(
        project_info=get_clozapine_project_info(),
        eligible_prediction_times_frame=clozapine_pred_times(),
        feature_specs=specs,
        feature_set_name="clozapine_text_feature_set",
        n_workers=None,
        step_size=datetime.timedelta(days=365),
        do_dataset_description=False,
    )
