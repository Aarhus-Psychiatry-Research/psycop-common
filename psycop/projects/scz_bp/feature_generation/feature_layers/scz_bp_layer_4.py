from collections.abc import Sequence

from timeseriesflattener.aggregation_fns import count
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
)

from psycop.common.feature_generation.loaders.raw.load_medications import (
    benzodiazepine_related_sleeping_agents,
    benzodiazepines,
    clozapine,
    first_gen_antipsychotics,
    lamotrigine,
    lithium,
    pregabaline,
    second_gen_antipsychotics,
    selected_nassa,
    snri,
    ssri,
    tca,
    valproate,
)
from psycop.projects.scz_bp.feature_generation.feature_layers.scz_bp_feature_layer import (
    SczBpFeatureLayer,
)


class SczBpLayer4(SczBpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[AnySpec]:
        layer = 4

        psychiatric_medications = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=first_gen_antipsychotics(),
                    name=f"first_gen_antipsychotics_layer_{layer}",
                ),
                NamedDataframe(
                    df=second_gen_antipsychotics(),
                    name=f"second_gen_antipsychotics_layer_{layer}",
                ),
                NamedDataframe(df=clozapine(), name=f"clozapine_layer_{layer}"),
                NamedDataframe(df=lithium(), name=f"lithium_layer_{layer}"),
                NamedDataframe(df=valproate(), name=f"valproate_layer_{layer}"),
                NamedDataframe(df=lamotrigine(), name=f"lamotrigine_layer_{layer}"),
                NamedDataframe(
                    df=benzodiazepines(),
                    name=f"benzodiazepines_layer_{layer}",
                ),
                NamedDataframe(df=pregabaline(), name=f"pregabaline_layer_{layer}"),
                NamedDataframe(df=ssri(), name=f"ssri_layer_{layer}"),
                NamedDataframe(df=snri(), name=f"snri_layer_{layer}"),
                NamedDataframe(df=tca(), name=f"tca_layer_{layer}"),
                NamedDataframe(
                    df=selected_nassa(),
                    name=f"selected_nassa_layer_{layer}",
                ),
                NamedDataframe(
                    df=benzodiazepine_related_sleeping_agents(),
                    name=f"benzodiazepine_related_sleeping_agents_layer_{layer}",
                ),
            ),
            lookbehind_days=lookbehind_days,
            aggregation_fns=[count],
            fallback=[0],
        ).create_combinations()

        return psychiatric_medications
