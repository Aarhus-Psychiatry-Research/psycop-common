from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from timeseriesflattener.aggregation_fns import boolean, mean, summed
from timeseriesflattener.df_transforms import df_with_multiple_values_to_named_dataframes
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec
from timeseriesflattener.feature_specs.single_specs import AnySpec

from psycop.common.global_utils.paths import TEXT_EMBEDDINGS_DIR
from psycop.projects.scz_bp.feature_generation.scz_bp_specify_features import SczBpFeatureSpecifier


class SczBpTextExperimentFeatures(SczBpFeatureSpecifier):
    def _text_specs_from_embedding_df(
        self, embedded_text_df: pd.DataFrame, name_prefix: str, lookbehind_days: list[float]
    ) -> Sequence[AnySpec]:
        embedded_text_df = df_with_multiple_values_to_named_dataframes(
            df=embedded_text_df,
            entity_id_col_name="dw_ek_borger",
            timestamp_col_name="timestamp",
            name_prefix=name_prefix,
        )  # type: ignore

        return PredictorGroupSpec(
            named_dataframes=embedded_text_df,  # type: ignore
            aggregation_fns=[mean],
            lookbehind_days=lookbehind_days,
            fallback=[np.nan],
        ).create_combinations()

    def _get_feature_by_note_type_and_model_name(
        self, note_type: str, model_name: str, lookbehind_days: list[float]
    ) -> Sequence[AnySpec]:
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
    ) -> list[AnySpec]:
        feature_specs: list[Sequence[AnySpec]] = [
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

    def get_keyword_specs(self, lookbehind_days: list[float]) -> list[AnySpec]:
        filename = "pse_keyword_counts_all_sfis.parquet"
        embedded_text_df = pd.read_parquet(TEXT_EMBEDDINGS_DIR / filename)
        if "overskrift" in embedded_text_df.columns:
            embedded_text_df = embedded_text_df.drop("overskrift", axis="columns")

        embedded_text_df = df_with_multiple_values_to_named_dataframes(
            df=embedded_text_df,
            entity_id_col_name="dw_ek_borger",
            timestamp_col_name="timestamp",
            name_prefix="pse_keywords_",
        )  # type: ignore

        return PredictorGroupSpec(
            named_dataframes=embedded_text_df,  # type: ignore
            aggregation_fns=[summed, boolean],
            lookbehind_days=lookbehind_days,
            fallback=[np.nan],
        ).create_combinations()
