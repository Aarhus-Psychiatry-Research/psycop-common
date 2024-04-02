"""Text-feature specification module."""
import logging
from typing import Union

import numpy as np
import pandas as pd
from timeseriesflattener.v1.aggregation_fns import mean
from timeseriesflattener.v1.feature_specs.single_specs import OutcomeSpec, PredictorSpec, StaticSpec

from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR
from psycop.projects.restraint.feature_generation.modules.specify_features import FeatureSpecifier

log = logging.getLogger(__name__)


class TextFeatureSpecifier(FeatureSpecifier):
    """Specify features based on prediction time."""

    def _get_text_specs(
        self, lookbehind_days: float, embedded_text: pd.DataFrame, spec_name: str, prefix: str
    ) -> PredictorSpec:
        return PredictorSpec(
            timeseries_df=embedded_text,
            feature_base_name=spec_name,
            lookbehind_days=lookbehind_days,
            aggregation_fn=mean,
            fallback=np.nan,
            prefix=prefix,
        )

    def _get_text_specs_by_type_and_model(
        self, note_type: str, model_name: str, lookbehind_days: float, spec_name: str
    ) -> PredictorSpec:
        filename = f"text_embeddings_{note_type}_{model_name}.parquet"
        embedded_text = pd.read_parquet(TEXT_EMBEDDINGS_DIR / filename)
        if "overskrift" in embedded_text.columns:
            embedded_text = embedded_text.drop("overskrift")

        return self._get_text_specs(
            embedded_text=embedded_text,
            prefix=f"{note_type}_{model_name}_",
            lookbehind_days=lookbehind_days,
            spec_name=spec_name,
        )

    def get_text_feature_specs(
        self, note_types: list[str], model_names: list[str]
    ) -> list[Union[StaticSpec, PredictorSpec, OutcomeSpec]]:
        """Generate text predictor spec list."""
        log.info("-------- Generating text predictor specs --------")

        if self.min_set_for_debug:
            return [
                self._get_text_specs_by_type_and_model(
                    lookbehind_days=10,
                    note_type="aktuelt_psykisk",
                    model_name="tfidf-500",
                    spec_name="test",
                )
            ]

        text_features: list[PredictorSpec] = []

        for note_type in note_types:
            for model_name in model_names:
                text_features += [
                    self._get_text_specs_by_type_and_model(
                        note_type=note_type,
                        model_name=model_name,
                        lookbehind_days=730,
                        spec_name="test",
                    )
                ]

        return self._get_static_predictor_specs() + text_features + self._get_outcome_specs()
