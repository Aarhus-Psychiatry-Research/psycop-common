"""Text-feature specification module."""
import logging
from typing import Callable

import numpy as np
from timeseriesflattener.aggregation_fns import mean
from timeseriesflattener.df_transforms import (
    df_with_multiple_values_to_named_dataframes,
)
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec
from timeseriesflattener.feature_specs.single_specs import PredictorSpec

from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.loaders.raw.load_embedded_text import (
    EmbeddedTextLoader,
)

log = logging.getLogger(__name__)


class TextFeatureSpecifier:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_sentence_transformer_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        log.info("-------- Generating sentence transformer specs --------")
        embedded_text_filename = (
            "text_embeddings_paraphrase-multilingual-MiniLM-L12-v2.parquet"
        )
        TEXT_SFIS = [
            "Observation af patient, Psykiatri",
            "Samtale med behandlingssigte",
            "Aktuelt psykisk",
            "Aktuelt socialt, Psykiatri",
            "Aftaler, Psykiatri",
            "Aktuelt somatisk, Psykiatri",
            "Objektivt psykisk",
            "Kontaktårsag",
            "Telefonnotat",
            "Semistruktureret diagnostisk interview",
            "Vurdering/konklusion",
        ]

        embedded_text = EmbeddedTextLoader.load_embedded_text(
            filename=embedded_text_filename,
            text_sfi_names=TEXT_SFIS,
            include_sfi_name=False,
            n_rows=None,
        ).to_pandas()

        embedded_text = df_with_multiple_values_to_named_dataframes(
            df=embedded_text,
            entity_id_col_name="dw_ek_borger",
            timestamp_col_name="timestamp",
            name_prefix="pred_sent_",
        )

        text_specs = PredictorGroupSpec(
            named_dataframes=embedded_text,
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[np.nan],
        ).create_combinations()

        return text_specs

    def _get_text_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("-------- Generating text predictor specs --------")

        if self.min_set_for_debug:
            return []

        text_specs = self._get_sentence_transformer_specs(
            resolve_multiple=[mean],
            interval_days=[7],
        )

        return text_specs

    def get_text_feature_specs(self) -> list[PredictorSpec]:
        """Get a spec set."""

        if self.min_set_for_debug:
            log.warning(
                "--- !!! Using the minimum set of features for debugging !!! ---",
            )
        return self._get_text_specs()
