from collections.abc import Sequence

from timeseriesflattener import PredictorGroupSpec
from timeseriesflattener.v1.aggregation_fns import boolean
from timeseriesflattener.v1.feature_specs.group_specs import NamedDataframe

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
from psycop.projects.bipolar.feature_generation.feature_layers.bp_feature_layer import (
    BpFeatureLayer,
)
from psycop.projects.bipolar.feature_generation.feature_layers.value_specification import (
    ValueSpecification,
)


class BpLayer4(BpFeatureLayer):
    def get_features(self, lookbehind_days: list[float]) -> Sequence[ValueSpecification]:
        layer = 4

        psychiatric_medications = list(
            PredictorGroupSpec(
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
                    NamedDataframe(df=benzodiazepines(), name=f"benzodiazepines_layer_{layer}"),
                    NamedDataframe(df=pregabaline(), name=f"pregabaline_layer_{layer}"),
                    NamedDataframe(df=ssri(), name=f"ssri_layer_{layer}"),
                    NamedDataframe(df=snri(), name=f"snri_layer_{layer}"),
                    NamedDataframe(df=tca(), name=f"tca_layer_{layer}"),
                    NamedDataframe(df=selected_nassa(), name=f"selected_nassa_layer_{layer}"),
                    NamedDataframe(
                        df=benzodiazepine_related_sleeping_agents(),
                        name=f"benzodiazepine_related_sleeping_agents_layer_{layer}",
                    ),
                ),
                lookbehind_days=lookbehind_days,
                aggregation_fns=[boolean],
                fallback=[0],
                entity_id_col_name_out="dw_ek_borger",
            ).create_combinations()
        )

        return psychiatric_medications
