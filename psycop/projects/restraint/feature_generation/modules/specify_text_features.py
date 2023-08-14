"""Text-feature specification module."""
import logging
from typing import Callable

import numpy as np
from timeseriesflattener.aggregation_fns import (
    mean_number_of_characters,  # type: ignore
    type_token_ratio,  # type: ignore
)
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    PredictorSpec,  # type: ignore
    TextPredictorSpec,  # type: ignore
)
from timeseriesflattener.text_embedding_functions import (  # type: ignore
    sklearn_embedding,
)

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

    def _get_text_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[TextPredictorSpec]:
        """Get bow all sfis specs"""
        log.info("-------- Generating tf-idf sfis specs --------")

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

    def get_feature_specs(
        self,
    ) -> list[TextPredictorSpec]:
        """Generate text predictor spec list."""
        log.info("-------- Generating text predictor specs --------")

        if self.min_set_for_debug:
            return self._get_text_specs(
                resolve_multiple=[concatenate],
                interval_days=[7],
            )  # type: ignore

        text_features = self._get_text_specs(
            resolve_multiple=[concatenate],
            interval_days=[30.0, 180.0, 365.0, 730.0],
        )

        return text_features
