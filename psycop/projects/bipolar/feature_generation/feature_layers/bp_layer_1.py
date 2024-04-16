from collections.abc import Sequence

import numpy as np
from timeseriesflattener import (
    PredictorGroupSpec,
    StaticFrame,
    StaticSpec,
    TimeDeltaSpec,
    TimestampValueFrame,
)
from timeseriesflattener.v1.aggregation_fns import count
from timeseriesflattener.v1.feature_specs.group_specs import NamedDataframe

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays, sex_female
from psycop.common.feature_generation.loaders.raw.load_medications import (
    antidepressives,
    antipsychotics,
)
from psycop.projects.bipolar.feature_generation.feature_layers.bp_feature_layer import (
    BpFeatureLayer,
)
from psycop.projects.bipolar.feature_generation.feature_layers.value_specification import (
    ValueSpecification,
)


class BpLayer1(BpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[ValueSpecification]:
        layer = 1

        sex_spec = [
            StaticSpec(
                value_frame=StaticFrame(
                    init_df=sex_female().rename({"value": "sex_female"}),
                    entity_id_col_name="dw_ek_borger",
                ),
                fallback=0,
                column_prefix=f"pred_layer_{layer}",
            )
        ]

        antipsychotics_spec = list(
            PredictorGroupSpec(
                named_dataframes=(
                    NamedDataframe(df=antipsychotics(), name=f"antipsychotics_layer_{layer}"),
                ),
                lookbehind_days=lookbehind_days,
                aggregation_fns=[count],
                fallback=[0],
            ).create_combinations()
        )

        antidepressives_spec = list(
            PredictorGroupSpec(
                named_dataframes=(
                    NamedDataframe(df=antidepressives(), name=f"antidepressives_layer_{layer}"),
                ),
                lookbehind_days=lookbehind_days,
                aggregation_fns=[count],
                fallback=[0],
            ).create_combinations()
        )

        age_spec = [
            TimeDeltaSpec(
                init_frame=TimestampValueFrame(
                    init_df=birthdays(),
                    entity_id_col_name="dw_ek_borger",
                    value_timestamp_col_name="date_of_birth",
                ),
                fallback=np.nan,
                output_name=f"layer_{layer}_age",
                time_format="years",
            )
        ]

        return sex_spec + antipsychotics_spec + antidepressives_spec + age_spec
