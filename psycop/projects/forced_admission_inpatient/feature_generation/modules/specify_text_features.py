"""Text-feature specification module."""
import logging
from typing import Callable, Union

import numpy as np
from timeseriesflattener.aggregation_fns import (
    concatenate,
    mean_number_of_characters,
    type_token_ratio,
)
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
    TextPredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    PredictorSpec,
    TextPredictorSpec,
)
from timeseriesflattener.text_embedding_functions import sklearn_embedding

from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.loaders.raw.load_text import load_aktuel_psykisk
from psycop.common.feature_generation.text_models.utils import load_text_model

log = logging.getLogger(__name__)


class TextFeatureSpecifier:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_text_features_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get mean character length sfis specs"""
        log.info("-------- Generating mean character length all sfis specs --------")

        text_features = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=load_aktuel_psykisk(), name="aktuelt_psykisk"),
            ),
            lookbehind_days=interval_days,
            fallback=[np.nan],
            aggregation_fns=resolve_multiple,
        ).create_combinations()

        return text_features

    def _get_text_embedding_features_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[TextPredictorSpec]:
        """Get bow all sfis specs"""
        log.info("-------- Generating bow all sfis specs --------")

        tfidf_model = load_text_model(
            filename="tfidf_psycop_train_all_sfis_preprocessed_sfi_type_Aktueltpsykisk_ngram_range_12_max_df_10_min_df_1_max_features_500.pkl",
        )

        tfidf_specs = TextPredictorGroupSpec(
            named_dataframes=[
                NamedDataframe(df=load_aktuel_psykisk(), name="aktuel_psykisk"),
            ],
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            embedding_fn_name="tfidf",
            fallback=[np.nan],
            embedding_fn=[sklearn_embedding],
            embedding_fn_kwargs=[{"model": tfidf_model}],
        ).create_combinations()

        return tfidf_specs

    def get_text_feature_specs(
        self,
    ) -> list[Union[TextPredictorSpec, PredictorSpec]]:
        """Generate text predictor spec list."""
        log.info("-------- Generating text predictor specs --------")

        if self.min_set_for_debug:
            text_embedding_features = self._get_text_embedding_features_specs(
                resolve_multiple=[concatenate],
                interval_days=[7, 30],
            )

            return text_embedding_features + self._get_text_features_specs(
                resolve_multiple=[mean_number_of_characters],
                interval_days=[7],
            )  # type: ignore

        text_features = self._get_text_features_specs(
            resolve_multiple=[mean_number_of_characters, type_token_ratio],
            interval_days=[7],
        )

        text_embedding_features = self._get_text_embedding_features_specs(
            resolve_multiple=[concatenate],
            interval_days=[7, 30],
        )

        return text_features + text_embedding_features
