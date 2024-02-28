import datetime as dt
from typing import Union

import numpy as np
import polars as pl
from timeseriesflattener import (
    BooleanOutcomeSpec,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
    TimeDeltaSpec,
    TimestampValueFrame,
    ValueFrame,
)
from timeseriesflattener.aggregators import HasValuesAggregator, MeanAggregator, SumAggregator

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR
from psycop.projects.scz_bp.feature_generation.scz_bp_specify_features import SczBpFeatureSpecifier

ValueSpecification = Union[
    PredictorSpec, OutcomeSpec, BooleanOutcomeSpec, TimeDeltaSpec, StaticSpec
]


def make_timedeltas_from_zero(lookbehind_days: list[float]) -> list[dt.timedelta]:
    return [dt.timedelta(days=lookbehind_day) for lookbehind_day in lookbehind_days]


class SczBpTextExperimentFeatures(SczBpFeatureSpecifier):
    def _text_specs_from_embedding_df(
        self, embedded_text_df: pl.DataFrame, name_prefix: str, lookbehind_days: list[float]
    ) -> list[ValueSpecification]:
        return [
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=embedded_text_df,
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookbehind_distances=make_timedeltas_from_zero(lookbehind_days=lookbehind_days),
                aggregators=[MeanAggregator()],
                fallback=np.nan,
                column_prefix=f"pred_{name_prefix}",
            )
        ]

    def _get_feature_by_note_type_and_model_name(
        self, note_type: str, model_name: str, lookbehind_days: list[float]
    ) -> list[ValueSpecification]:
        filename = f"text_embeddings_{note_type}_{model_name}.parquet"
        embedded_text_df = pl.read_parquet(TEXT_EMBEDDINGS_DIR / filename)
        if "overskrift" in embedded_text_df.columns:
            embedded_text_df = embedded_text_df.drop("overskrift")

        return self._text_specs_from_embedding_df(
            embedded_text_df=embedded_text_df,
            name_prefix=f"{note_type}_{model_name}_",
            lookbehind_days=lookbehind_days,
        )

    def get_age_spec(self) -> list[ValueSpecification]:
        return [
            TimeDeltaSpec(
                init_frame=TimestampValueFrame(
                    init_df=birthdays(),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="date_of_birth",
                ),
                fallback=np.nan,
                output_name="age_in_years",
                time_format="years",
            )
        ]

    def get_feature_specs(  # type: ignore[override]
        self, lookbehind_days: list[float], note_type: str, model_name: str
    ) -> list[ValueSpecification]:
        feature_specs = [self._get_metadata_specs(), self._get_outcome_specs()]
        feature_specs.append(
            self._get_feature_by_note_type_and_model_name(
                note_type=note_type, model_name=model_name, lookbehind_days=lookbehind_days
            )
        )

        # flatten the sequence of lists
        features = [feature for sublist in feature_specs for feature in sublist]
        return features

    def get_keyword_specs(
        self, lookbehind_days: list[float], df: pl.DataFrame | None
    ) -> list[ValueSpecification]:
        if df is None:
            filename = "pse_keyword_counts_all_sfis.parquet"
            df = pl.read_parquet(TEXT_EMBEDDINGS_DIR / filename)
            if "overskrift" in df.columns:
                df = df.drop("overskrift")

        return [
            PredictorSpec(
                value_frame=ValueFrame(
                    init_df=df,
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",
                ),
                lookbehind_distances=make_timedeltas_from_zero(lookbehind_days=lookbehind_days),
                aggregators=[HasValuesAggregator(), SumAggregator()],
                fallback=np.nan,
                column_prefix="pred_pse_keywords",
            )
        ]
