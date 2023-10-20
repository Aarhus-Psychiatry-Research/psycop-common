from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence
from venv import create
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
    boolean,
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

AnySpecType = AnySpec | PredictorSpec | OutcomeSpec | StaticSpec

class FeatureLayer(Protocol):
    def get_features(self, lookbehind_days: int) -> Sequence[PredictorSpec]:
        ...

@dataclass(frozen=True)
class LayerNamedDataframe(NamedDataframe):
    layer: int

    @property
    def name(self) -> str:
        return f"{self.name}_layer_{self.layer}"