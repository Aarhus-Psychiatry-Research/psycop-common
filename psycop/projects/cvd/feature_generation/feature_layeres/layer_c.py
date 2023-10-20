from psycop.common.feature_generation.loaders.raw.load_diagnoses import f0_disorders, f1_disorders, f2_disorders, f3_disorders, f4_disorders, f5_disorders, f6_disorders, f7_disorders, f8_disorders, f9_disorders
from psycop.common.feature_generation.loaders.raw.load_lab_results import hdl
from psycop.common.feature_generation.loaders.raw.load_medications import top_10_weight_gaining_antipsychotics
from psycop.projects.cvd.feature_generation.feature_layeres.base import AnySpecType, FeatureLayer, LayerNamedDataframe, LayerPosition


from timeseriesflattener.aggregation_fns import boolean, count
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec


from typing import Sequence


class CVDLayerC(FeatureLayer):
    def get_features(self, lookbehind_days: int) -> Sequence[AnySpecType]:
        psychiatric_disorders = PredictorGroupSpec(
            named_dataframes=(
                LayerNamedDataframe(df=f0_disorders(), name="f0_disorders_layer", layer=layer),
                LayerNamedDataframe(df=f1_disorders(), name="f1_disorders", layer=layer),
                LayerNamedDataframe(df=f2_disorders(), name="f2_disorders", layer=layer),
                LayerNamedDataframe(df=f3_disorders(), name="f3_disorders", layer=layer),
                LayerNamedDataframe(df=f4_disorders(), name="f4_disorders", layer=layer),
                LayerNamedDataframe(df=f5_disorders(), name="f5_disorders", layer=layer),
                LayerNamedDataframe(df=f6_disorders(), name="f6_disorders", layer=layer),
                LayerNamedDataframe(df=f7_disorders(), name="f7_disorders", layer=layer),
                LayerNamedDataframe(df=f8_disorders(), name="f8_disorders", layer=layer),
                LayerNamedDataframe(df=f9_disorders(), name="f9_disorders", layer=layer),
                LayerNamedDataframe(
                    df=top_10_weight_gaining_antipsychotics(),
                    name="top_10_weight_gaining_antipsychotics",
                    layer=layer,
                ),
            ),
            aggregation_fns=[count],
            lookbehind_days=[lookbehind_days],
            fallback=[0],
        ).create_combinations()

        antipsychotics_spec = PredictorGroupSpec(
            named_dataframes=(
                LayerNamedDataframe(df=top_10_weight_gaining_antipsychotics(), name="antipsychotics", layer=layer),),
                lookbehind_days=[lookbehind_days],
                aggregation_fns=[boolean],
                fallback=[0],
        ).create_combinations()

        hdl_spec = PredictorGroupSpec(
            named_dataframes=[LayerNamedDataframe(df=hdl(), name="hdl", layer=layer)],
            lookbehind_days=[lookbehind_days],
            aggregation_fns=[boolean],
            fallback=[0],
        ).create_combinations()

        return psychiatric_disorders + antipsychotics_spec+ hdl_spec