"""Text-feature specification module."""
import logging

import numpy as np
from timeseriesflattener.aggregation_fns import AggregationFunType, mean
from timeseriesflattener.df_transforms import df_with_multiple_values_to_named_dataframes
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec
from timeseriesflattener.feature_specs.single_specs import PredictorSpec  # type: ignore

from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw.load_embedded_text import EmbeddedTextLoader

log = logging.getLogger(__name__)


class TextFeatureSpecifier:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_text_specs(
        self,
        resolve_multiple: list[AggregationFunType],
        interval_days: list[float],
        embedded_text_filename: str,
    ) -> list[PredictorSpec]:
        log.info("-------- Generating tf-idf specs --------")

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
            name_prefix="sent_",
        )

        tfidf_specs = PredictorGroupSpec(
            named_dataframes=embedded_text,
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[np.nan],
        ).create_combinations()

        return tfidf_specs

    def get_feature_specs(self) -> list[PredictorSpec]:
        """Generate text predictor spec list."""
        log.info("-------- Generating text predictor specs --------")

        if self.min_set_for_debug:
            return self._get_text_specs(
                resolve_multiple=[mean],
                interval_days=[1],
                embedded_text_filename="text_tfidf.parquet",
            )  # type: ignore

        resolve_multiple = [mean]
        interval_days = [60.0, 365.0, 730.0]

        tfidf_features = self._get_text_specs(
            resolve_multiple=resolve_multiple,
            interval_days=interval_days,
            embedded_text_filename="text_tfidf.parquet",
        )

        self._get_text_specs(
            resolve_multiple=resolve_multiple,
            interval_days=interval_days,
            embedded_text_filename="text_embeddings_paraphrase-multilingual-MiniLM-L12-v2.parquet",
        )

        return tfidf_features
