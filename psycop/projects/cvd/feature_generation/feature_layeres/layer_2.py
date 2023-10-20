import numpy as np
from timeseriesflattener.feature_specs.single_specs import PredictorSpec
from psycop.common.feature_generation.loaders.raw.load_diagnoses import acute_myocardial_infarction, chronic_lung_disease, f0_disorders, f1_disorders, f2_disorders, f3_disorders, f4_disorders, f5_disorders, f6_disorders, f7_disorders, f8_disorders, f9_disorders, ischemic_stroke
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c, hdl
from psycop.common.feature_generation.loaders.raw.load_medications import top_10_weight_gaining_antipsychotics
from psycop.common.feature_generation.loaders.raw.load_procedures import cabg, pad, pci
from psycop.projects.cvd.feature_generation.feature_layeres.base import AnySpecType, FeatureLayer, LayerNamedDataframe, LayerPosition


from timeseriesflattener.aggregation_fns import boolean, count, mean
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec


from typing import Sequence


class CVDLayer2(FeatureLayer):
    def get_features(self, lookbehind_days: int) -> Sequence[PredictorSpec]:
        layer=2
        specs = PredictorGroupSpec(
            lookbehind_days=[lookbehind_days],
            named_dataframes=[
                LayerNamedDataframe(
                    df=acute_myocardial_infarction(), name="acute_myocardial_infarction", layer=layer
                ),
                LayerNamedDataframe(df=pci(), name="pci", layer=layer),
                LayerNamedDataframe(df=cabg(), name="cabg", layer=layer),
                LayerNamedDataframe(df=ischemic_stroke(), name="ischemic_stroke", layer=layer),
                LayerNamedDataframe(df=pad(), name="peripheral_arterial_disease", layer=layer),
                LayerNamedDataframe(df=hba1c(), name="hba1c", layer=layer),
                LayerNamedDataframe(df=chronic_lung_disease(), name="chronic_lung_disease", layer=layer),
            ],
            aggregation_fns=[mean],
            fallback=[np.nan],
        ).create_combinations()

        return specs