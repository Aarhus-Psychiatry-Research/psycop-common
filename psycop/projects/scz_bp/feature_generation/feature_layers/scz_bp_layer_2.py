from collections.abc import Sequence

from timeseriesflattener.aggregation_fns import count
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
)

from psycop.common.feature_generation.loaders.raw.load_visits import (
    admissions,
    physical_visits_to_psychiatry,
    physical_visits_to_somatic,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_feature_layer import (
    SczBpFeatureLayer,
)


class SczBpLayer2(SczBpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[AnySpec]:
        layer = 2

        visits_to_psychiatry_spec = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=physical_visits_to_psychiatry(
                        return_value_as_visit_length_days=False,
                    ),
                    name=f"physical_visits_to_psychiatry_layer_{layer}",
                ),
            ),
            lookbehind_days=lookbehind_days,
            aggregation_fns=[count],
            fallback=[0],
        ).create_combinations()

        visits_to_somatic_spec = PredictorGroupSpec(
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

        admissions_to_psychiatry_spec = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=admissions(shak_code=6600, shak_sql_operator="="),
                    name=f"antidepressives_layer_{layer}",
                ),
            ),
            lookbehind_days=lookbehind_days,
            aggregation_fns=[count],
            fallback=[0],
        ).create_combinations()

        return (
            visits_to_psychiatry_spec
            + visits_to_somatic_spec
            + admissions_to_psychiatry_spec
        )
