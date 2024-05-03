"""Feature specification module."""
import datetime as dt
import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from timeseriesflattener import (
    BooleanOutcomeSpec,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticFrame,
    StaticSpec,
    TimeDeltaSpec,
    TimestampValueFrame,
    ValueFrame,
)
from timeseriesflattener.aggregators import CountAggregator, MaxAggregator
from timeseriesflattener.flattener import ValueSpecification
from timeseriesflattener.v1.aggregation_fns import boolean
from timeseriesflattener.v1.feature_specs.group_specs import NamedDataframe

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import (
    birthdays,
    sex_female,
)
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
from psycop.common.feature_generation.loaders.raw.load_visits import (
    admissions,
    get_time_of_first_visit_to_psychiatry,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_1 import (
    SczBpLayer1,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_2 import (
    SczBpLayer2,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_3 import (
    SczBpLayer3,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_layer_4 import (
    SczBpLayer4,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.bp_diagnoses import (
    get_first_bp_diagnosis,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.first_scz_or_bp_diagnosis import (
    get_diagnosis_type_of_first_scz_bp_diagnosis,
    get_time_of_first_scz_or_bp_diagnosis,
)
from psycop.projects.scz_bp.feature_generation.outcome_specification.scz_diagnoses import (
    get_first_scz_diagnosis,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)


SczBpFeatureLayers = {1: SczBpLayer1, 2: SczBpLayer2, 3: SczBpLayer3, 4: SczBpLayer4}


def make_timedeltas_from_zero(look_days: list[float]) -> list[dt.timedelta]:
    return [dt.timedelta(days=lookbehind_day) for lookbehind_day in look_days]


class SczBpMetaDataFeatureSpecifier:
    """Feature specification class."""

    def _get_outcome_specs(self) -> list[OutcomeSpec]:
        log.info("-------- Generating outcome specs --------")

        return [
            OutcomeSpec(
                value_frame=ValueFrame(
                    init_df=SczBpCohort.get_outcome_timestamps().frame.rename(
                        {"value": "first_scz_or_bp"}
                    ),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookahead_distances=make_timedeltas_from_zero(
                    look_days=[year * 365 for year in (1, 2, 3, 4, 5)]
                ),
                fallback=0,
                aggregators=[MaxAggregator()],
            )
        ]

    def _get_diagnoses_specs(self) -> list[PredictorSpec]:
        psychiatric_diagnoses = list(
            PredictorGroupSpec(
                named_dataframes=(
                    NamedDataframe(df=f0_disorders(), name=f"f0_disorders"),
                    NamedDataframe(df=f1_disorders(), name=f"f1_disorders"),
                    NamedDataframe(df=f2_disorders(), name=f"f2_disorders"),
                    NamedDataframe(df=f3_disorders(), name=f"f3_disorders"),
                    NamedDataframe(df=f4_disorders(), name=f"f4_disorders"),
                    NamedDataframe(df=f5_disorders(), name=f"f5_disorders"),
                    NamedDataframe(df=f6_disorders(), name=f"f6_disorders"),
                    NamedDataframe(df=f7_disorders(), name=f"f7_disorders"),
                    NamedDataframe(df=f8_disorders(), name=f"f8_disorders"),
                    NamedDataframe(df=f9_disorders(), name=f"f9_disorders"),
                ),
                lookbehind_days=[730],
                aggregation_fns=[boolean],
                fallback=[0],
                entity_id_col_name_out="dw_ek_borger"
            ).create_combinations()
        )
        return psychiatric_diagnoses

    def _get_metadata_specs(self) -> list[ValueSpecification]:
        log.info("-------- Generating metadata specs --------")

        return [
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=get_diagnosis_type_of_first_scz_bp_diagnosis().rename(
                        {"source": "scz_or_bp_indicator"}
                    ),
                    entity_id_col_name="dw_ek_borger",
                ),
                column_prefix="meta",
                fallback=np.nan,
            ),
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=get_time_of_first_scz_or_bp_diagnosis().rename(
                        {"timestamp": "time_of_diagnosis"}
                    ),
                    entity_id_col_name="dw_ek_borger",
                ),
                column_prefix="meta",
                fallback=np.nan,
            ),
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=pl.from_pandas(get_first_scz_diagnosis())
                    .rename({"timestamp": "time_of_scz_diagnosis"})
                    .drop("value"),
                    entity_id_col_name="dw_ek_borger",
                ),
                column_prefix="meta",
                fallback=np.nan,
            ),
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=pl.from_pandas(get_first_bp_diagnosis())
                    .rename({"timestamp": "time_of_bp_diagnosis"})
                    .drop("value"),
                    entity_id_col_name="dw_ek_borger",
                ),
                column_prefix="meta",
                fallback=np.nan,
            ),
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=get_time_of_first_visit_to_psychiatry().rename(
                        {"timestamp": "first_visit"}
                    ),
                    entity_id_col_name="dw_ek_borger",
                ),
                column_prefix="meta",
                fallback=np.nan,
            ),
            BooleanOutcomeSpec(
                init_frame=TimestampValueFrame(
                    init_df=pl.from_pandas(get_first_scz_diagnosis()).drop("value"),
                    entity_id_col_name="dw_ek_borger",
                ),
                lookahead_distances=[dt.timedelta(days=365 * 5)],
                aggregators=[MaxAggregator()],
                output_name="scz_diagnosis",
                column_prefix="meta",
            ),
            BooleanOutcomeSpec(
                init_frame=TimestampValueFrame(
                    init_df=pl.from_pandas(get_first_bp_diagnosis()).drop("value"),
                    entity_id_col_name="dw_ek_borger",
                ),
                lookahead_distances=[dt.timedelta(days=365 * 5)],
                aggregators=[MaxAggregator()],
                output_name="bp_diagnosis",
                column_prefix="meta",
            ),
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=sex_female().rename({"value": "sex_female"}),
                    entity_id_col_name="dw_ek_borger",
                ),
                fallback=0,
                column_prefix="meta",
            ),
            TimeDeltaSpec(
                init_frame=TimestampValueFrame(
                    init_df=birthdays(),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="date_of_birth",
                ),
                fallback=np.nan,
                output_name="age",
                column_prefix="meta",
                time_format="years",
            ),
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=pl.from_pandas(
                        admissions(shak_code=6600, shak_sql_operator="=")
                    ).rename({"value": "n_admissions"}),
                    entity_id_col_name="dw_ek_borger",
                ),
                lookbehind_distances=[dt.timedelta(days=730)],
                aggregators=[CountAggregator()],
                fallback=0,
                column_prefix="meta",  # rename
            ),
        ]

    def get_feature_specs(self) -> list[ValueSpecification]:
        feature_specs: list[Sequence[ValueSpecification]] = [
            self._get_metadata_specs(),
            self._get_outcome_specs(),
            self._get_diagnoses_specs()
        ]

        # Flatten the Sequence of lists
        features = [feature for sublist in feature_specs for feature in sublist]
        return features


def get_scz_bp_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="scz_bp", project_path=OVARTACI_SHARED_DIR / "scz_bp")


if __name__ == "__main__":
    generate_feature_set(
        project_info=get_scz_bp_project_info(),
        eligible_prediction_times_frame=SczBpCohort.get_filtered_prediction_times_bundle().prediction_times,
        feature_specs=SczBpMetaDataFeatureSpecifier().get_feature_specs(),
        n_workers=None,
        do_dataset_description=False,
        feature_set_name="metadata_for_table1",
    )
