"""Text-feature specification module."""
import datetime as dt
import logging
from typing import Union

import numpy as np
import pandas as pd
from timeseriesflattener import (
    OutcomeSpec,
    PredictorSpec,
    StaticFrame,
    StaticSpec,
    TimeDeltaSpec,
    TimestampValueFrame,
    ValueFrame,
)
from timeseriesflattener.aggregators import HasValuesAggregator, MeanAggregator

from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw import sql_load
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR
from psycop.projects.restraint.cohort.restraint_cohort_definer import RestraintCohortDefiner

log = logging.getLogger(__name__)


class TextFeatureSpecifier:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_static_predictor_specs(self) -> list[StaticSpec | TimeDeltaSpec]:
        """Get static predictor specs."""
        return [
            StaticSpec(
                value_frame=StaticFrame(init_df=sex_female(), entity_id_col_name="dw_ek_borger"),
                fallback=np.nan,
                column_prefix=self.project_info.prefix.predictor,
            ),
            TimeDeltaSpec(
                init_frame=TimestampValueFrame(
                    init_df=birthdays(),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="date_of_birth",
                ),
                fallback=np.nan,
                output_name="age",
                time_format="years",
                column_prefix=self.project_info.prefix.predictor,
            ),
        ]

    def _get_outcome_specs(self) -> list[OutcomeSpec]:
        return [
            OutcomeSpec(
                value_frame=ValueFrame(
                    sql_load(
                    "SELECT *, 1 as value FROM fct.psycop_coercion_outcome_timestamps_2"
                    )
                    # RestraintCohortDefiner.get_outcome_timestamps()
                    # .frame.to_pandas()
                    # .assign(value=1)
                    .rename(columns={"first_mechanical_restraint": "timestamp"})
                    .dropna(subset="timestamp"),
                    entity_id_col_name="dw_ek_borger",
                ),
                lookahead_distances=[dt.timedelta(days=2)],
                aggregators=[HasValuesAggregator()],
                fallback=0,
                column_prefix="outc_",
            )
        ]

    def _get_text_specs(
        self, lookbehind_days: dt.timedelta, embedded_text: pd.DataFrame, prefix: str
    ) -> PredictorSpec:
        return PredictorSpec(
            value_frame=ValueFrame(
                init_df=embedded_text,
                entity_id_col_name="dw_ek_borger",
                value_timestamp_col_name="timestamp",
            ),
            lookbehind_distances=[lookbehind_days],
            aggregators=[MeanAggregator()],
            fallback=np.nan,
            column_prefix=f"pred_{prefix}",
        )

    def _get_text_specs_by_type_and_model(
        self, note_type: str, model_name: str, lookbehind_days: dt.timedelta
    ) -> PredictorSpec:
        filename = f"text_embeddings_{note_type}_{model_name}.parquet"
        embedded_text = pd.read_parquet(TEXT_EMBEDDINGS_DIR / filename)
        if "overskrift" in embedded_text.columns:
            embedded_text = embedded_text.drop(columns="overskrift")

        return self._get_text_specs(
            embedded_text=embedded_text,
            prefix=f"{note_type}_{model_name}_",
            lookbehind_days=lookbehind_days,
        )

    def get_text_feature_specs(
        self, note_type: str, model_name: str
    ) -> list[Union[OutcomeSpec, PredictorSpec, StaticSpec, TimeDeltaSpec]]:
        """Generate text predictor spec list."""
        log.info("-------- Generating text predictor specs --------")

        if self.min_set_for_debug:
            return [
                self._get_text_specs_by_type_and_model(
                    lookbehind_days=dt.timedelta(days=10),
                    note_type="aktuelt_psykisk",
                    model_name="tfidf-500",
                )
            ]

        text_features: list[PredictorSpec] = []

        text_features += [
            self._get_text_specs_by_type_and_model(
                note_type=note_type, model_name=model_name, lookbehind_days=dt.timedelta(days=730)
            )
        ]

        return self._get_static_predictor_specs() + text_features + self._get_outcome_specs()
