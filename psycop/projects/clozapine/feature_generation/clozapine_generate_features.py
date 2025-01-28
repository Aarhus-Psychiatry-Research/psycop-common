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
    MeanAggregator,
    SumAggregator,
    UniqueCountAggregator,
)

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.clozapine.feature_generation.cohort_definition.clozapine_cohort_definition import (
    clozapine_pred_times,
)
from psycop.projects.clozapine.feature_generation.cohort_definition.outcome_specification.combine_text_structured_clozapine_outcome import (
    combine_structured_and_text_outcome,
)
from psycop.projects.clozapine.loaders.coercion import skema_1, skema_2_without_nutrition, skema_3
from psycop.projects.clozapine.loaders.demographics import birthdays, sex_female
from psycop.projects.clozapine.loaders.diagnoses import (
    cluster_b,
    f0_disorders,
    f1_disorders,
    f3_disorders,
    f4_disorders,
    f5_disorders,
    f6_disorders,
    f7_disorders,
    f8_disorders,
    f9_disorders,
    manic_and_bipolar,
)
from psycop.projects.clozapine.loaders.ect import ect_all
from psycop.projects.clozapine.loaders.lab_results import (
    cancelled_standard_lab_results,
    p_aripiprazol,
    p_clozapine,
    p_ethanol,
    p_haloperidol,
    p_olanzapine,
    p_paliperidone,
    p_paracetamol,
    p_risperidone,
)
from psycop.projects.clozapine.loaders.medications import (
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
from psycop.projects.clozapine.loaders.structured_sfi import (
    broeset_violence_checklist,
    suicide_risk_assessment,
)
from psycop.projects.clozapine.loaders.visits import (
    admissions_clozapine_2024,
    emergency_visits_clozapine_2024,
    physical_visits_to_psychiatry_clozapine_2024,
    physical_visits_to_somatic_clozapine_2024,
)

TEXT_FILE_NAME = "not_labelled_yet_clozapine_text.parquet"


def make_timedeltas_from_zero(look_days: list[float]) -> list[datetime.timedelta]:
    return [datetime.timedelta(days=lookbehind_day) for lookbehind_day in look_days]


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
        "demographic": [
            ts.OutcomeSpec(
                value_frame=ts.ValueFrame(
                    init_df=combine_structured_and_text_outcome(),
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
        "contacts": [
            ContinuousSpec(
                physical_visits_to_psychiatry_clozapine_2024,
                aggregation_fns=[CountAggregator()],
                fallback=0,
            ),
            ContinuousSpec(
                physical_visits_to_somatic_clozapine_2024,
                aggregation_fns=[CountAggregator()],
                fallback=0,
            ),
            ContinuousSpec(
                partial(admissions_clozapine_2024, shak_code=6600, shak_sql_operator="="),
                aggregation_fns=[CountAggregator()],
                fallback=0,
                name_overwrite="admissions_to_psychiatry",
            ),
            ContinuousSpec(
                partial(
                    admissions_clozapine_2024,
                    shak_code=6600,
                    shak_sql_operator="=",
                    return_value_as_visit_length_days=True,
                ),
                aggregation_fns=[SumAggregator()],
                fallback=0,
                name_overwrite="admissions_to_psychiatry_duration",
            ),
            ContinuousSpec(
                partial(admissions_clozapine_2024, shak_code=6600, shak_sql_operator="!="),
                aggregation_fns=[CountAggregator()],
                fallback=0,
                name_overwrite="admissions_not_to_psychiatry",
            ),
            ContinuousSpec(
                partial(
                    admissions_clozapine_2024,
                    shak_code=6600,
                    shak_sql_operator="!=",
                    return_value_as_visit_length_days=True,
                ),
                aggregation_fns=[SumAggregator()],
                fallback=0,
                name_overwrite="admissions_not_to_psychiatry_duration",
            ),
            ContinuousSpec(
                partial(emergency_visits_clozapine_2024, shak_code=6600, shak_sql_operator="="),
                aggregation_fns=[CountAggregator()],
                fallback=0,
                name_overwrite="emergency_visits_to_psychiatry",
            ),
            ContinuousSpec(
                partial(emergency_visits_clozapine_2024, shak_code=6600, shak_sql_operator="!="),
                aggregation_fns=[CountAggregator()],
                fallback=0,
                name_overwrite="emergency_visits_not_to_psychiatry",
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
        "diagnoses": [
            BooleanSpec(f0_disorders),
            BooleanSpec(f1_disorders),
            BooleanSpec(manic_and_bipolar),
            BooleanSpec(f3_disorders),
            BooleanSpec(f4_disorders),
            BooleanSpec(f5_disorders),
            BooleanSpec(f6_disorders),
            BooleanSpec(cluster_b),
            BooleanSpec(f7_disorders),
            BooleanSpec(f8_disorders),
            BooleanSpec(f9_disorders),
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
        "lab-results": [
            BooleanSpec(cancelled_standard_lab_results),
            BooleanSpec(p_aripiprazol),
            BooleanSpec(p_clozapine),
            BooleanSpec(p_ethanol),
            BooleanSpec(p_haloperidol),
            BooleanSpec(p_olanzapine),
            BooleanSpec(p_paliperidone),
            BooleanSpec(p_paracetamol),
            BooleanSpec(p_risperidone),
        ],
        "coercion": [
            # coercion loaders return duration of coercion
            ContinuousSpec(skema_1, aggregation_fns=[MeanAggregator()], fallback=0),
            ContinuousSpec(
                skema_2_without_nutrition, aggregation_fns=[MeanAggregator()], fallback=0
            ),
            ContinuousSpec(skema_3, aggregation_fns=[MeanAggregator()], fallback=0),
        ],
        "ect": [
            # coercion loaders return duration of coercion
            ContinuousSpec(ect_all, aggregation_fns=[MeanAggregator()], fallback=0)
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
        feature_set_name="clozapine_full_feature_set_without_text",
        n_workers=None,
        step_size=datetime.timedelta(days=365),
        do_dataset_description=False,
    )
