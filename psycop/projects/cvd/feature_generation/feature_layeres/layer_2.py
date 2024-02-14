from collections.abc import Sequence

import numpy as np
from timeseriesflattener.aggregation_fns import AggregationFunType
from timeseriesflattener.feature_specs.group_specs import NamedDataframe, PredictorGroupSpec
from timeseriesflattener.feature_specs.single_specs import PredictorSpec

from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    acute_myocardial_infarction,
    chronic_lung_disease,
    ischemic_stroke,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c
from psycop.common.feature_generation.loaders.raw.load_procedures import cabg, pad, pci
from psycop.projects.cvd.feature_generation.feature_layeres.base import FeatureLayer


class CVDLayer2(FeatureLayer):
    def get_features(
        self, lookbehind_days: Sequence[int], aggregation_fns: Sequence[AggregationFunType]
    ) -> Sequence[PredictorSpec]:
        layer = 2
        specs = PredictorGroupSpec(
            lookbehind_days=lookbehind_days,
            named_dataframes=[
                NamedDataframe(
                    df=acute_myocardial_infarction(),
                    name=f"acute_myocardial_infarction_layer_{layer}",
                ),
                NamedDataframe(df=pci(), name=f"pci_layer_{layer}"),
                NamedDataframe(df=cabg(), name=f"cabg_layer_{layer}"),
                NamedDataframe(df=ischemic_stroke(), name=f"ischemic_stroke_layer_{layer}"),
                # NamedDataframe(df=pad(), name=f"peripheral_arterial_disease_layer_{layer}"),
                NamedDataframe(df=hba1c(), name=f"hba1c_layer_{layer}"),
                NamedDataframe(
                    df=chronic_lung_disease(), name=f"chronic_lung_disease_layer_{layer}"
                ),
            ],
            aggregation_fns=aggregation_fns,
            fallback=[np.nan],
        ).create_combinations()

        return specs
