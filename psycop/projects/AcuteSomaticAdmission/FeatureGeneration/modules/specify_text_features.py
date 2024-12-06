"""Text-feature specification module."""

import logging
from typing import Callable, Literal

import numpy as np
from timeseriesflattener.v1.aggregation_fns import mean
from timeseriesflattener.v1.df_transforms import df_with_multiple_values_to_named_dataframes
from timeseriesflattener.v1.feature_specs.group_specs import PredictorGroupSpec
from timeseriesflattener.v1.feature_specs.single_specs import PredictorSpec

from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.feature_generation.loaders.raw.load_embedded_text import EmbeddedTextLoader

log = logging.getLogger(__name__)


class TextFeatureSpecifier:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_sentence_transformer_specs(
        self,
        resolve_multiple: list[Callable],  # type: ignore
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        log.info("-------- Generating sentence transformer specs --------")
        embedded_text_filename = "text_embeddings_paraphrase-multilingual-MiniLM-L12-v2"
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
            embedding_view_name=embedded_text_filename,
            text_sfi_names=TEXT_SFIS,
            include_sfi_name=False,
            n_rows=None,
        )

        embedded_text = df_with_multiple_values_to_named_dataframes(
            df=embedded_text,
            entity_id_col_name="dw_ek_borger",
            timestamp_col_name="timestamp",
            name_prefix="pred_sent_",
        )

        sentence_transformer_specs = PredictorGroupSpec(
            named_dataframes=embedded_text,
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[np.nan],
        ).create_combinations()

        return sentence_transformer_specs

    def _get_tfidf_specs(
        self,
        resolve_multiple: list[Callable],  # type: ignore
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        log.info("-------- Generating tfidf specs --------")
        embedded_text_filename = "text_train_val_test_tfidf_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet"
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
            embedding_view_name=embedded_text_filename,
            text_sfi_names=TEXT_SFIS,
            include_sfi_name=False,
            n_rows=None,
        )

        embedded_text = df_with_multiple_values_to_named_dataframes(
            df=embedded_text,
            entity_id_col_name="dw_ek_borger",
            timestamp_col_name="timestamp",
            name_prefix="pred_",
        )

        tfidf_specs = PredictorGroupSpec(
            named_dataframes=embedded_text,
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[np.nan],
        ).create_combinations()

        return tfidf_specs

    def _get_text_specs(
        self, embedding_method: Literal["tfidf", "sentence_transformer", "both"] = "tfidf"
    ) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("-------- Generating text predictor specs --------")

        if self.min_set_for_debug:
            return []

        sentence_transformer_specs = self._get_sentence_transformer_specs(
            resolve_multiple=[mean], interval_days=[180]
        )

        tfidf_specs = self._get_tfidf_specs(resolve_multiple=[mean], interval_days=[180])

        if embedding_method == "both":
            return sentence_transformer_specs + tfidf_specs

        if embedding_method == "sentence_transformer":
            return sentence_transformer_specs

        return tfidf_specs

    def get_text_feature_specs(
        self, embedding_method: Literal["tfidf", "sentence_transformer", "both"] = "tfidf"
    ) -> list[PredictorSpec]:
        """Get a spec set."""

        if self.min_set_for_debug:
            log.warning("--- !!! Using the minimum set of features for debugging !!! ---")
        return self._get_text_specs(embedding_method=embedding_method)
