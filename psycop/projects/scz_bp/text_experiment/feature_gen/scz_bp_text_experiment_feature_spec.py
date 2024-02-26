import datetime as dt
from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np
import pandas as pd
from timeseriesflattener import PredictorSpec, ValueFrame, OutcomeSpec, BooleanOutcomeSpec, TimeDeltaSpec, StaticSpec
from timeseriesflattener.aggregators import HasValuesAggregator, MeanAggregator, SumAggregator
    HasValuesAggregator,
    MeanAggregator,
    SumAggregator,
)
from timeseriesflattener.v1.aggregation_fns import mean
from timeseriesflattener.v1.df_transforms import (
    df_with_multiple_values_to_named_dataframes,
)
from timeseriesflattener.v1.feature_specs.group_specs import PredictorGroupSpec
from timeseriesflattener.v1.feature_specs.single_specs import AnySpec

from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR
from psycop.projects.scz_bp.feature_generation.scz_bp_specify_features import (
    SczBpFeatureSpecifier,
)

ValueSpecification = Union[
    PredictorSpec, OutcomeSpec, BooleanOutcomeSpec, TimeDeltaSpec, StaticSpec
]

def make_timedeltas_from_zero(lookbehind_days: list[float]) -> list[dt.timedelta]:
    return [dt.timedelta(days=lookbehind_day) for lookbehind_day in lookbehind_days]

class SczBpTextExperimentFeatures(SczBpFeatureSpecifier):
    def _text_specs_from_embedding_df(
        self, embedded_text_df: pd.DataFrame, name_prefix: str, lookbehind_days: list[float]
    ) -> list[ValueSpecification]:
        return [PredictorSpec(
            value_frame=ValueFrame(
                init_df=embedded_text_df, entity_id_col_name="dw_ek_borger", value_timestamp_col_name="timestamp"
            ),
            lookbehind_distances=make_timedeltas_from_zero(lookbehind_days=lookbehind_days),
            aggregators=[MeanAggregator()],
            fallback=np.nan,
            column_prefix=f"pred_{name_prefix}"
        )]

    def _get_feature_by_note_type_and_model_name(
        self, note_type: str, model_name: str, lookbehind_days: list[float]
    ) -> list[ValueSpecification]:
        filename = f"text_embeddings_{note_type}_{model_name}.parquet"
        embedded_text_df = pd.read_parquet(TEXT_EMBEDDINGS_DIR / filename)
        if "overskrift" in embedded_text_df.columns:
            embedded_text_df = embedded_text_df.drop("overskrift", axis="columns")

        return self._text_specs_from_embedding_df(
            embedded_text_df=embedded_text_df,
            name_prefix=f"{note_type}_{model_name}_",
            lookbehind_days=lookbehind_days,
        )

    def get_feature_specs(  # type: ignore[override]
        self, lookbehind_days: list[float], note_type: str, model_name: str
    ) -> list[ValueSpecification]:
        feature_specs = [
            self._get_metadata_specs(),
            self._get_outcome_specs(),
        ]
        feature_specs.append(
            self._get_feature_by_note_type_and_model_name(
                note_type=note_type, model_name=model_name, lookbehind_days=lookbehind_days
            )
        )

        # flatten the sequence of lists
        features = [feature for sublist in feature_specs for feature in sublist]
        return features

    def get_keyword_specs(self, lookbehind_days: list[float]) -> list[ValueSpecification]:
        filename = "pse_keyword_counts_all_sfis.parquet"
        embedded_text_df = pd.read_parquet(TEXT_EMBEDDINGS_DIR / filename)
        if "overskrift" in embedded_text_df.columns:
            embedded_text_df = embedded_text_df.drop("overskrift", axis="columns")

        return [PredictorSpec(
            value_frame=ValueFrame(
                init_df=embedded_text_df, entity_id_col_name="dw_ek_borger", value_timestamp_col_name="timestamp"
            ),
            lookbehind_distances=make_timedeltas_from_zero(lookbehind_days=lookbehind_days),
            aggregators=[HasValuesAggregator(), SumAggregator()],
            fallback=np.nan,
            column_prefix=f"pred_pse_keywords"
        )]
