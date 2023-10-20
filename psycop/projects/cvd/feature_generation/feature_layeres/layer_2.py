from collections.abc import Sequence

import numpy as np
from timeseriesflattener.aggregation_fns import mean
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec
from timeseriesflattener.feature_specs.single_specs import PredictorSpec

from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    acute_myocardial_infarction,
    chronic_lung_disease,
    ischemic_stroke,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c
from psycop.common.feature_generation.loaders.raw.load_procedures import cabg, pad, pci
from psycop.projects.cvd.feature_generation.feature_layeres.base import (
    FeatureLayer,
    LayerNamedDataframe,
)


class CVDLayer2(FeatureLayer):
    def get_features(self, lookbehind_days: int) -> Sequence[PredictorSpec]:
        layer = 2
        specs = PredictorGroupSpec(
            lookbehind_days=[lookbehind_days],
            named_dataframes=[
                LayerNamedDataframe(
                    df=acute_myocardial_infarction(),
                    name="acute_myocardial_infarction",
                    layer=layer,
                ),
                LayerNamedDataframe(df=pci(), name="pci", layer=layer),
                LayerNamedDataframe(df=cabg(), name="cabg", layer=layer),
                LayerNamedDataframe(
                    df=ischemic_stroke(),
                    name="ischemic_stroke",
                    layer=layer,
                ),
                LayerNamedDataframe(
                    df=pad(),
                    name="peripheral_arterial_disease",
                    layer=layer,
                ),
                LayerNamedDataframe(df=hba1c(), name="hba1c", layer=layer),
                LayerNamedDataframe(
                    df=chronic_lung_disease(),
                    name="chronic_lung_disease",
                    layer=layer,
                ),
            ],
            aggregation_fns=[mean],
            fallback=[np.nan],
        ).create_combinations()

        return specs
