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
from timeseriesflattener.aggregators import (
    HasValuesAggregator,
    LatestAggregator,
    MeanAggregator,
    UniqueCountAggregator,
)

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.clozapine.text_models.clozapine_text_model_paths import TEXT_EMBEDDINGS_DIR
from psycop.projects.psychometrics.hamilton.cohort.hamilton_cohort_definition import (
    hamilton_outcome_timestamps,
    hamilton_pred_times,
)
from psycop.projects.psychometrics.loaders.coercion import (
    skema_1,
    skema_2_without_nutrition,
    skema_3,
)
from psycop.projects.psychometrics.loaders.demographics import birthdays, sex_female
from psycop.projects.psychometrics.loaders.medications import (
    alcohol_abstinence,
    analgesic,
    analgesic_fast,
    antidepressives,
    antidepressives_fast,
    antipsychotics_fast,
    anxiolytics,
    anxiolytics_fast,
    aripiprazole_depot,
    benzodiazepine_related_sleeping_agents,
    benzodiazepines,
    first_gen_antipsychotics,
    haloperidol_depot,
    hyperactive_disorders_medications,
    hypnotics,
    lithium,
    olanzapine,
    olanzapine_depot,
    opioid_dependence,
    paliperidone_depot,
    perphenazine_depot,
    risperidone_depot,
    second_gen_antipsychotics,
    zuclopenthixol_depot,
)
from psycop.projects.psychometrics.loaders.structured_sfi import (
    broeset_violence_checklist,
    suicide_risk_assessment,
)

TEXT_FILE_NAME = "XX.parquet"


def make_timedeltas_from_zero(look_days: list[float]) -> list[datetime.timedelta]:
    return [datetime.timedelta(days=lookbehind_day) for lookbehind_day in look_days]


def get_hamilton_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="hamilton", project_path=OVARTACI_SHARED_DIR / "hamilton")


def split_df_to_list(  # noqa: D417
    df: pl.DataFrame, entity_id_col_name: str = "entity_id", timestamp_col_name: str = "timestamp"
) -> list[pl.DataFrame]:
    """
    Splits a Polars DataFrame into multiple DataFrames, each containing the entity and
    timestamp columns along with one value column.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame to be split.
        entity_id_col_name (str): The name of the column containing entity IDs.
        timestamp_col_name (str): The name of the column containing timestamps.

    Returns:
        list[pl.DataFrame]: A list of Polars DataFrames, each with the entity ID column, timestamp column, and a single additional value column.
    """
    mandatory_cols = [entity_id_col_name, timestamp_col_name]

    # Validate required columns
    for col in mandatory_cols:
        if col not in df.columns:
            raise ValueError(f"Missing mandatory column: '{col}'")

    value_cols = [col for col in df.columns if col not in mandatory_cols]

    return [df.select([*mandatory_cols, col]) for col in value_cols]


def _init_clozapine_predictor(
    df_loader: Callable[[], pd.DataFrame],
    layer: str,
    fallback: float | int,
    aggregation_fns: Sequence[ts.aggregators.Aggregator],
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [10, 20, 30]
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
    pred_times = hamilton_pred_times()

    df_tfidf_split = split_df_to_list(
        df=pl.read_parquet(TEXT_EMBEDDINGS_DIR / TEXT_FILE_NAME).drop("overskrift"),
        entity_id_col_name="dw_ek_borger",
        timestamp_col_name="timestamp",
    )

    feature_layers = {
        "demographic": [
            ts.OutcomeSpec(
                value_frame=ts.ValueFrame(
                    init_df=hamilton_outcome_timestamps().frame,
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookahead_distances=make_timedeltas_from_zero(
                    look_days=[year * 365 for year in (1, 2, 3, 4, 5)]
                ),
                aggregators=[ts.MaxAggregator()],
                fallback=0,
                column_prefix="outc_clozapine",
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
        "selvmord-broset": [
            ContinuousSpec(
                suicide_risk_assessment,
                aggregation_fns=[
                    LatestAggregator(timestamp_col_name="timestamp"),
                    MeanAggregator(),
                ],
                fallback=np.NaN,
            ),
            ContinuousSpec(
                broeset_violence_checklist,
                aggregation_fns=[LatestAggregator("timestamp"), MeanAggregator()],
                fallback=np.NaN,
            ),
        ],
        "medication": [
            BooleanSpec(first_gen_antipsychotics),
            BooleanSpec(second_gen_antipsychotics),
            BooleanSpec(lithium),
            BooleanSpec(anxiolytics),
            BooleanSpec(antidepressives),
            BooleanSpec(hyperactive_disorders_medications),
            BooleanSpec(benzodiazepines),
            BooleanSpec(hypnotics),
            BooleanSpec(olanzapine),
            BooleanSpec(analgesic),
            BooleanSpec(alcohol_abstinence),
            BooleanSpec(opioid_dependence),
            BooleanSpec(benzodiazepine_related_sleeping_agents),
        ],
        "unique_count_medication": [
            ContinuousSpec(
                antipsychotics_fast, aggregation_fns=[UniqueCountAggregator()], fallback=0
            ),
            ContinuousSpec(
                antidepressives_fast, aggregation_fns=[UniqueCountAggregator()], fallback=0
            ),
            ContinuousSpec(anxiolytics_fast, aggregation_fns=[UniqueCountAggregator()], fallback=0),
            ContinuousSpec(analgesic_fast, aggregation_fns=[UniqueCountAggregator()], fallback=0),
        ],
        "depot-medication": [
            BooleanSpec(aripiprazole_depot),
            BooleanSpec(haloperidol_depot),
            BooleanSpec(olanzapine_depot),
            BooleanSpec(paliperidone_depot),
            BooleanSpec(perphenazine_depot),
            BooleanSpec(risperidone_depot),
            BooleanSpec(zuclopenthixol_depot),
        ],
        "coercion": [
            # coercion loaders return duration of coercion
            ContinuousSpec(skema_1, aggregation_fns=[MeanAggregator()], fallback=0),
            ContinuousSpec(
                skema_2_without_nutrition, aggregation_fns=[MeanAggregator()], fallback=0
            ),
            ContinuousSpec(skema_3, aggregation_fns=[MeanAggregator()], fallback=0),
        ],
        "layer_text": [
            ts.PredictorSpec(
                value_frame=ts.ValueFrame(
                    init_df=df,
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookbehind_distances=[datetime.timedelta(days=180)],
                aggregators=[MeanAggregator()],
                fallback=np.nan,
                column_prefix=f"pred_layer_text__{df.columns[-1]}",
            )
            for df in df_tfidf_split
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
        project_info=get_hamilton_project_info(),
        eligible_prediction_times_frame=hamilton_pred_times(),
        feature_specs=specs,
        feature_set_name="hamilton_full_feature_set",
        n_workers=None,
        step_size=datetime.timedelta(days=365),
        do_dataset_description=False,
    )
