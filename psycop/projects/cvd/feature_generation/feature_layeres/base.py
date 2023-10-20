from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence
from timeseriesflattener.feature_specs.group_specs import PredictorGroupSpec

from timeseriesflattener.feature_specs.single_specs import AnySpec
import logging

import numpy as np
from timeseriesflattener.aggregation_fns import (
    AggregationFunType,
    count,
    latest,
    maximum,
    mean,
    minimum,
)
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    OutcomeGroupSpec,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    BaseModel,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)

from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    SCORE2_CVD,
    acute_myocardial_infarction,
    chronic_lung_disease,
    essential_hypertension,
    f0_disorders,
    f1_disorders,
    f2_disorders,
    f3_disorders,
    f4_disorders,
    f5_disorders,
    f6_disorders,
    f7_disorders,
    f8_disorders,
    f9_disorders,
    gerd,
    hyperlipidemia,
    ischemic_stroke,
    polycystic_ovarian_syndrome,
    sleep_apnea,
)
from psycop.common.feature_generation.loaders.raw.load_lab_results import (
    alat,
    albumine_creatinine_ratio,
    arterial_p_glc,
    crp,
    egfr,
    fasting_ldl,
    fasting_p_glc,
    hba1c,
    hdl,
    ldl,
    ogtt,
    scheduled_glc,
    triglycerides,
    unscheduled_p_glc,
    urinary_glc,
)
from psycop.common.feature_generation.loaders.raw.load_medications import (
    antihypertensives,
    antipsychotics,
    benzodiazepine_related_sleeping_agents,
    benzodiazepines,
    clozapine,
    diuretics,
    gerd_drugs,
    lamotrigine,
    lithium,
    pregabaline,
    selected_nassa,
    snri,
    ssri,
    statins,
    tca,
    top_10_weight_gaining_antipsychotics,
    valproate,
)
from psycop.common.feature_generation.loaders.raw.load_procedures import cabg, pad, pci
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    bmi,
    height_in_cm,
    weight_in_kg,
)
from psycop.projects.t2d.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_diabetes_indicator,
)
from psycop.projects.t2d.feature_generation.cohort_definition.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)

class LayerPosition(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"

AnySpecType = AnySpec | PredictorSpec | OutcomeSpec | StaticSpec

class FeatureLayer(Protocol):
    def get_features(self, layer: LayerPosition, lookbehind_days: int) -> Sequence[AnySpec]:
        ...

@dataclass(frozen=True)
class LayerNamedDataframe(NamedDataframe):
    layer: str

    @property
    def name(self) -> str:
        return f"{self.name}_layer_{self.layer}"
    

class LayerC(FeatureLayer):
    def get_features(self, layer: LayerPosition, lookbehind_days: int) -> Sequence[AnySpecType]:
        psychiatric_disorders = PredictorGroupSpec(
            named_dataframes=(
                LayerNamedDataframe(df=f0_disorders(), name="f0_disorders_layer", layer=layer.value),
                LayerNamedDataframe(df=f1_disorders(), name="f1_disorders", layer=layer.value),
                LayerNamedDataframe(df=f2_disorders(), name="f2_disorders", layer=layer.value),
                LayerNamedDataframe(df=f3_disorders(), name="f3_disorders", layer=layer.value),
                LayerNamedDataframe(df=f4_disorders(), name="f4_disorders", layer=layer.value),
                LayerNamedDataframe(df=f5_disorders(), name="f5_disorders", layer=layer.value),
                LayerNamedDataframe(df=f6_disorders(), name="f6_disorders", layer=layer.value),
                LayerNamedDataframe(df=f7_disorders(), name="f7_disorders", layer=layer.value),
                LayerNamedDataframe(df=f8_disorders(), name="f8_disorders", layer=layer.value),
                LayerNamedDataframe(df=f9_disorders(), name="f9_disorders", layer=layer.value),
                LayerNamedDataframe(
                    df=top_10_weight_gaining_antipsychotics(),
                    name="top_10_weight_gaining_antipsychotics",
                    layer=layer.value,
                ),
            ),
            aggregation_fns=[count],
            lookbehind_days=[lookbehind_days],
            fallback=[0],
        ).create_combinations()

        hdl_spec = PredictorGroupSpec(
            named_dataframes=[LayerNamedDataframe(df=hdl(), name="hdl", layer=layer.value)],
            lookbehind_days=[lookbehind_days],
            aggregation_fns=[count],
            fallback=[0],
        ).create_combinations()

        return psychiatric_disorders + hdl_spec