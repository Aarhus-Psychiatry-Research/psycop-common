"""Feature specification module."""
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
from psycop.projects.cvd.feature_generation.feature_layeres.base import FeatureLayer, LayerPosition
from psycop.projects.cvd.feature_generation.feature_layeres.layer_1 import CVDLayer1
from psycop.projects.cvd.feature_generation.feature_layeres.layer_2 import CVDLayer2
from psycop.projects.cvd.feature_generation.feature_layeres.layer_3 import CVDLayer3
from psycop.projects.t2d.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_diabetes_indicator,
)
from psycop.projects.t2d.feature_generation.cohort_definition.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)

log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[AnySpec]


class FeatureSpecifier:
    """Feature specification class."""

    def __init__(
        self,
        project_info: ProjectInfo,
    ) -> None:
        self.project_info = project_info

    def _get_static_predictor_specs(self) -> list[StaticSpec]:
        """Get static predictor specs."""
        return [
            StaticSpec(
                feature_base_name="sex_female",
                timeseries_df=sex_female(),
                prefix=self.project_info.prefix.predictor,
            ),
        ]

    def _get_outcome_specs(self) -> list[OutcomeSpec]:
        """Get outcome specs."""
        log.info("-------- Generating outcome specs --------")

        return OutcomeGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=SCORE2_CVD(),
                    name="score2_cvd",
                ),
            ],
            lookahead_days=[365 * 5],
            aggregation_fns=[maximum],
            fallback=[0],
            incident=[True],
            prefix=self.project_info.prefix.outcome,
        ).create_combinations()

    def get_feature_specs(self, layer: int) -> list[AnySpec]:
        """Get a spec set."""

        layers: list[FeatureLayer] = [CVDLayer1()]

        if layer >= 2:
            layers.append(CVDLayer2())
        if layer >= 3:
            layers.append(CVDLayer3())
        if layer > 3:
            raise ValueError(f"Layer {layer} not supported.")

        temporal_predictor_sequences = [layer.get_features(lookbehind_days=730) for layer in layers]
        temporal_predictors = [item for sublist in temporal_predictor_sequences for item in sublist]

        return (
            temporal_predictors
            + self._get_static_predictor_specs()
            + self._get_outcome_specs()
        )