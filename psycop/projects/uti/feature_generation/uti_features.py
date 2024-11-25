"""Main feature generation."""

import datetime
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd
import polars as pl
import timeseriesflattener as ts
from timeseriesflattener.aggregators import HasValuesAggregator

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.uti.feature_generation.cohort_definition.uti_cohort_definer import (
    uti_pred_times,
)
from psycop.projects.uti.feature_generation.outcome_definition.uti_outcomes import (
    uti_outcome_timestamps,
    uti_postive_urine_sample_outcome_timestamps,
    uti_relevant_antibiotics_administrations_outcome_timestamps,
)


def get_uti_project_info() -> ProjectInfo:
    return ProjectInfo(
        project_name="uti", project_path=OVARTACI_SHARED_DIR / "uti" / "flattened_datasets"
    )


def _init_uti_predictor(
    df_loader: Callable[[], pd.DataFrame],
    layer: str,
    fallback: float | int,
    aggregation_fns: Sequence[ts.aggregators.Aggregator],
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [1, 5]
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
            return _init_uti_predictor(
                df_loader=pair.spec.loader,
                layer=pair.layer,
                fallback=pair.spec.fallback,
                aggregation_fns=pair.spec.aggregation_fns,
                name_overwrite=pair.spec.name_overwrite,
            )
        case BooleanSpec():
            return _init_uti_predictor(
                df_loader=pair.spec.loader,
                layer=pair.layer,
                fallback=pair.spec.fallback,
                aggregation_fns=pair.spec.aggregation_fns,
                name_overwrite=pair.spec.name_overwrite,
            )
        case CategoricalSpec():
            return _init_uti_predictor(
                df_loader=pair.spec.loader,
                layer=pair.layer,
                fallback=pair.spec.fallback,
                aggregation_fns=pair.spec.aggregation_fns,
                name_overwrite=pair.spec.name_overwrite,
            )
        case ts.PredictorSpec() | ts.OutcomeSpec() | ts.StaticSpec() | ts.TimeDeltaSpec():
            return pair.spec


def uti_generate_features(
    outcomes: Literal["combined", "urine_samples", "antibiotics"],
    lookahead_days: int = 3,
    feature_set_name: str = "uti_outcomes_full_definition",
    add_feature_layers: dict[
        str, list[ts.PredictorSpec | ts.OutcomeSpec | ts.StaticSpec | ts.TimeDeltaSpec]
    ]
    | None = None,
):
    match outcomes:
        case "combined":
            outcome_df = uti_outcome_timestamps()
        case "urine_samples":
            outcome_df = uti_postive_urine_sample_outcome_timestamps()
        case "antibiotics":
            outcome_df = uti_relevant_antibiotics_administrations_outcome_timestamps()

    feature_layers = {
        "basic": [
            ts.OutcomeSpec(
                value_frame=ts.ValueFrame(
                    init_df=outcome_df,
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookahead_distances=[datetime.timedelta(days=lookahead_days)],
                aggregators=[ts.MaxAggregator()],
                fallback=0,
                column_prefix="outc_uti",
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
        ]
    }

    if add_feature_layers:
        for layer_name, layer_specs in add_feature_layers.items():
            feature_layers[layer_name] = layer_specs  # type: ignore

    layer_spec_pairs = [
        LayerSpecPair(layer, spec)
        for layer, spec_list in feature_layers.items()
        for spec in spec_list
    ]

    logging.info("Loading specifications")
    specs = [_pair_to_spec(layer_spec) for layer_spec in layer_spec_pairs]

    logging.info("Generating feature set")
    generate_feature_set(
        project_info=get_uti_project_info(),
        eligible_prediction_times_frame=uti_pred_times().prediction_times,
        feature_specs=specs,
        feature_set_name=feature_set_name,
        n_workers=None,
        step_size=datetime.timedelta(days=365),
        do_dataset_description=False,
    )


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(  # type: ignore
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        stream=sys.stdout,
    )
    uti_generate_features(
        outcomes="combined", lookahead_days=1, feature_set_name="uti_outcomes_full_definition"
    )
    uti_generate_features(
        outcomes="urine_samples", lookahead_days=1, feature_set_name="uti_outcomes_urine_samples"
    )

    uti_generate_features(
        outcomes="antibiotics", lookahead_days=1, feature_set_name="uti_outcomes_antibiotics"
    )
