from collections.abc import Sequence

from timeseriesflattener import PredictorGroupSpec
from timeseriesflattener.v1.aggregation_fns import count
from timeseriesflattener.v1.feature_specs.group_specs import NamedDataframe

from psycop.common.feature_generation.loaders.raw.load_visits import (
    admissions,
    physical_visits_to_psychiatry,
    physical_visits_to_somatic,
)
from psycop.projects.bipolar.feature_generation.feature_layers.bp_feature_layer import (
    BpFeatureLayer,
)
from psycop.projects.bipolar.feature_generation.feature_layers.value_specification import (
    ValueSpecification,
)


class BpLayer2(BpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[ValueSpecification]:
        layer = 2

        visits_to_psychiatry_spec = list(
            PredictorGroupSpec(
                named_dataframes=(
                    NamedDataframe(
                        df=physical_visits_to_psychiatry(return_value_as_visit_length_days=False),
                        name=f"physical_visits_to_psychiatry_layer_{layer}",
                    ),
                ),
                lookbehind_days=lookbehind_days,
                aggregation_fns=[count],
                fallback=[0],
            ).create_combinations()
        )

        visits_to_somatic_spec = list(
            PredictorGroupSpec(
                named_dataframes=(
                    NamedDataframe(
                        df=physical_visits_to_somatic(),
                        name=f"physical_visits_to_somatic_layer_{layer}",
                    ),
                ),
                lookbehind_days=lookbehind_days,
                aggregation_fns=[count],
                fallback=[0],
            ).create_combinations()
        )

        admissions_to_psychiatry_spec = list(
            PredictorGroupSpec(
                named_dataframes=(
                    NamedDataframe(
                        df=admissions(shak_code=6600, shak_sql_operator="="),
                        name=f"admissions_layer_{layer}",
                    ),
                ),
                lookbehind_days=lookbehind_days,
                aggregation_fns=[count],
                fallback=[0],
            ).create_combinations()
        )

        return visits_to_psychiatry_spec + visits_to_somatic_spec + admissions_to_psychiatry_spec
