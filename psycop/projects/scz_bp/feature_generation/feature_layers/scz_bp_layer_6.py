from collections.abc import Sequence

import numpy as np
from timeseriesflattener.aggregation_fns import mean
from timeseriesflattener.df_transforms import (
    df_with_multiple_values_to_named_dataframes,
)
from timeseriesflattener.feature_specs.group_specs import (
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
)

from psycop.common.feature_generation.loaders.raw.load_embedded_text import (
    EmbeddedTextLoader,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_feature_layer import (
    SczBpFeatureLayer,
)


class SczBpLayer6(SczBpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[AnySpec]:
        layer = 6

        tf_idf_filename = "text_train_val_test_tfidf_all_sfis_ngram_range_12_max_df_09_min_df_2_max_features_750.parquet"
        text_sfi_names = [
            "Observation af patient, Psykiatri",
            "Samtale med behandlingssigte",
            # "Aktuelt psykisk", in layer 5
            "Aktuelt socialt, Psykiatri",
            "Aftaler, Psykiatri",
            "Aktuelt somatisk, Psykiatri",
            "Objektivt psykisk",
            "Kontaktårsag",
            "Telefonnotat",
            "Semistruktureret diagnostisk interview",
            "Vurdering/konklusion",
        ]

        embedded_text_df = EmbeddedTextLoader.load_embedded_text(
            filename=tf_idf_filename,
            text_sfi_names=text_sfi_names,
            include_sfi_name=False,
            n_rows=None,
        ).to_pandas()

        embedded_text_df = df_with_multiple_values_to_named_dataframes(
            df=embedded_text_df,  # type: ignore
            entity_id_col_name="dw_ek_borger",
            timestamp_col_name="timestamp",
            name_prefix=f"pred_tf_idf_all_sfis_layer_{layer}_",
        )
        tfidf_all_text_sfis_specs = PredictorGroupSpec(
            named_dataframes=embedded_text_df,
            aggregation_fns=[mean],
            lookbehind_days=lookbehind_days,
            fallback=[np.nan],
        ).create_combinations()

        return tfidf_all_text_sfis_specs