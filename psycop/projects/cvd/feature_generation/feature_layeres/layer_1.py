import numpy as np
from psycop.common.feature_generation.loaders.raw.load_diagnoses import essential_hypertension, f0_disorders, f1_disorders, f2_disorders, f3_disorders, f4_disorders, f5_disorders, f6_disorders, f7_disorders, f8_disorders, f9_disorders
from psycop.common.feature_generation.loaders.raw.load_lab_results import hdl, ldl
from psycop.common.feature_generation.loaders.raw.load_medications import top_10_weight_gaining_antipsychotics
from psycop.projects.cvd.feature_generation.feature_layeres.base import AnySpecType, FeatureLayer, LayerNamedDataframe, LayerPosition


from timeseriesflattener.aggregation_fns import boolean, count, mean
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec, PredictorSpec


from typing import Sequence


class CVDLayer1(FeatureLayer):
    def get_features(self, lookbehind_days: int) -> Sequence[PredictorSpec]:
        layer = 1
        ldl_spec = PredictorGroupSpec(
                named_dataframes=(LayerNamedDataframe(df=ldl(), name="ldl", layer=layer),),
                aggregation_fns=[mean],
                fallback=[np.nan],
                lookbehind_days=[lookbehind_days],
            ).create_combinations()
        
        essential_hypertension_spec = PredictorGroupSpec(
                named_dataframes=(LayerNamedDataframe(df=essential_hypertension(), name="essential_hypertension", layer=layer),),
                aggregation_fns=[count],
                fallback=[0],
                lookbehind_days=[lookbehind_days],
            ).create_combinations()

        return ldl_spec + essential_hypertension_spec