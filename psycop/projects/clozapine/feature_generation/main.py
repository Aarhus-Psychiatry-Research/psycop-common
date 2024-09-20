"""Main feature generation."""

import datetime
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import timeseriesflattener as ts
from timeseriesflattener.aggregators import (
    CountAggregator,
    HasValuesAggregator,
    LatestAggregator,
    MaxAggregator,
    MeanAggregator,
    SumAggregator,
)

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw.load_coercion import skema_1, skema_2, skema_3
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
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
from psycop.common.feature_generation.loaders.raw.load_lab_results import (
    cancelled_standard_lab_results,
    crp,
    lymphocytes,
    triglycerides,
)
from psycop.common.feature_generation.loaders.raw.load_medications import (
    antidepressives,
    antipsychotics,
    benzodiazepine_related_sleeping_agents,
    benzodiazepines,
    first_gen_antipsychotics,
    lamotrigine,
    lithium,
    pregabaline,
    second_gen_antipsychotics,
    selected_nassa,
    snri,
    ssri,
    tca,
    valproate,
)
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    broeset_violence_checklist,
    suicide_risk_assessment,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    admissions,
    ambulatory_visits,
    emergency_visits,
    physical_visits_to_psychiatry,
    physical_visits_to_somatic,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR, TEXT_EMBEDDINGS_DIR
from psycop.projects.clozapine.feature_generation.cohort_definition.clozapine_cohort_definition import (
    clozapine_outcome_timestamps,
    clozapine_pred_times,
)

TEXT_FILE_NAME = "text_embeddings_all_relevant_tfidf-1000.parquet"


def get_clozapine_project_info() -> ProjectInfo:
    return ProjectInfo(
        project_name="clozapine", project_path=OVARTACI_SHARED_DIR / "clozapine" / "feature_set"
    )


def _init_clozapine_predictor(
    df_loader: Callable[[], pd.DataFrame],
    layer: str,
    fallback: float | int,
    aggregation_fns: Sequence[ts.aggregators.Aggregator],
    lookbehind_distances: Sequence[datetime.timedelta] = [
        datetime.timedelta(days=i) for i in [90, 182, 365]
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

    feature_layers = {
        "demographics": [
            ts.OutcomeSpec(
                value_frame=ts.ValueFrame(
                    init_df=clozapine_outcome_timestamps().frame, entity_id_col_name="dw_ek_borger"
                ),
                lookahead_distances=[datetime.timedelta(days=365)],
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
        "contacts": [
            ContinuousSpec(
                physical_visits_to_psychiatry, aggregation_fns=[CountAggregator()], fallback=0
            ),
            ContinuousSpec(
                physical_visits_to_somatic, aggregation_fns=[CountAggregator()], fallback=0
            ),
            ContinuousSpec(
                partial(admissions, shak_code=6600, shak_sql_operator="="),
                aggregation_fns=[CountAggregator()],
                fallback=0,
                name_overwrite="admissions_to_psychiatry",
            ),
            ContinuousSpec(
                partial(
                    admissions,
                    shak_code=6600,
                    shak_sql_operator="=",
                    return_value_as_visit_length_days=True,
                ),
                aggregation_fns=[SumAggregator()],
                fallback=0,
                name_overwrite="admissions_to_psychiatry_duration",
            ),
            ContinuousSpec(
                partial(admissions, shak_code=6600, shak_sql_operator="!="),
                aggregation_fns=[CountAggregator()],
                fallback=0,
                name_overwrite="admissions_not_to_psychiatry",
            ),
            ContinuousSpec(
                partial(
                    admissions,
                    shak_code=6600,
                    shak_sql_operator="!=",
                    return_value_as_visit_length_days=True,
                ),
                aggregation_fns=[SumAggregator()],
                fallback=0,
                name_overwrite="admissions_not_to_psychiatry_duration",
            ),
            ContinuousSpec(
                partial(ambulatory_visits, shak_code=6600, shak_sql_operator="="),
                aggregation_fns=[CountAggregator()],
                fallback=0,
                name_overwrite="ambulatory_visits_to_psychiatry",
            ),
            ContinuousSpec(
                partial(emergency_visits, shak_code=6600, shak_sql_operator="="),
                aggregation_fns=[CountAggregator()],
                fallback=0,
                name_overwrite="emergency_visits_to_psychiatry",
            ),
            ContinuousSpec(
                partial(emergency_visits, shak_code=6600, shak_sql_operator="!="),
                aggregation_fns=[CountAggregator()],
                fallback=0,
                name_overwrite="emergency_visits_not_to_psychiatry",
            ),
        ],
        "Medication overall": [
            ContinuousSpec(antipsychotics, aggregation_fns=[CountAggregator()], fallback=0),
            ContinuousSpec(antidepressives, aggregation_fns=[CountAggregator()], fallback=0),
            ContinuousSpec(benzodiazepines, aggregation_fns=[CountAggregator()], fallback=0),
        ],
        "broset + suicide risk assessment": [
            ContinuousSpec(
                broeset_violence_checklist,
                aggregation_fns=[LatestAggregator("timestamp"), MeanAggregator()],
                fallback=np.NaN,
            ),
            ContinuousSpec(
                suicide_risk_assessment,
                aggregation_fns=[
                    LatestAggregator(timestamp_col_name="timestamp"),
                    MaxAggregator(),
                    MeanAggregator(),
                ],
                fallback=np.NaN,
            ),
        ],
        "diagnoses": [
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
        ],
        "medication": [
            BooleanSpec(first_gen_antipsychotics),
            BooleanSpec(second_gen_antipsychotics),
            BooleanSpec(lithium),
            BooleanSpec(valproate),
            BooleanSpec(lamotrigine),
            BooleanSpec(pregabaline),
            BooleanSpec(ssri),
            BooleanSpec(snri),
            BooleanSpec(tca),
            BooleanSpec(selected_nassa),
            BooleanSpec(benzodiazepine_related_sleeping_agents),
        ],
        "coercion": [BooleanSpec(skema_1), BooleanSpec(skema_2), BooleanSpec(skema_3)],
        "lab tests": [
            ContinuousSpec(crp, aggregation_fns=[MeanAggregator()], fallback=np.NaN),
            ContinuousSpec(lymphocytes, aggregation_fns=[MeanAggregator()], fallback=np.NaN),
            ContinuousSpec(triglycerides, aggregation_fns=[MeanAggregator()], fallback=np.NaN),
            ContinuousSpec(
                cancelled_standard_lab_results, aggregation_fns=[MeanAggregator()], fallback=np.NaN
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
                lookbehind_distances=[datetime.timedelta(days=182)],
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
        feature_set_name="clozapine_feature_set_with_text",
        n_workers=None,
        step_size=datetime.timedelta(days=365),
        do_dataset_description=False,
    )
