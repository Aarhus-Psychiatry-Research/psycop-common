"""Text-feature specification module."""
import logging
import datetime as dt
from typing import Union
import pandas as pd

import numpy as np
from timeseriesflattener import (
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
    ValueFrame,
)
from timeseriesflattener.aggregators import MeanAggregator

from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw.load_embedded_text import EmbeddedTextLoader
from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR
from psycop.projects.restraint.feature_generation.modules.specify_features import FeatureSpecifier

log = logging.getLogger(__name__)



class TextFeatureSpecifier(FeatureSpecifier):
    """Specify features based on prediction time."""

    def _get_text_specs(
        self,
        lookbehind_days: dt.timedelta,
        embedded_text: pd.DataFrame,
        prefix: str,
    ) -> PredictorSpec:

        return PredictorSpec(
            value_frame=ValueFrame(init_df=embedded_text,
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="timestamp",),
            lookbehind_distances=[lookbehind_days],
            aggregators=[MeanAggregator()],
            fallback=np.nan,
            column_prefix=f"pred_{prefix}",
        )

    def _get_text_specs_by_type_and_model(
        self, note_type: str, model_name: str, lookbehind_days: dt.timedelta,
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

    def get_text_feature_specs(self, note_types: list[str], model_names: list[str]) -> list[PredictorSpec]: # list[Union[OutcomeSpec, PredictorSpec, StaticSpec]]:
        """Generate text predictor spec list."""
        log.info("-------- Generating text predictor specs --------")

        if self.min_set_for_debug:
            return [self._get_text_specs_by_type_and_model(
                lookbehind_days=dt.timedelta(days=10),
                note_type="aktuelt_psykisk",
                model_name="tfidf-500",
            )]

        text_features: list[PredictorSpec] = []

        for note_type in note_types:
            for model_name in model_names:
                text_features += [self._get_text_specs_by_type_and_model(
                    note_type=note_type,
                    model_name=model_name,
                    lookbehind_days=dt.timedelta(days=730),
                )]

        return text_features # self._get_static_predictor_specs() + text_features + self._get_outcome_specs()
