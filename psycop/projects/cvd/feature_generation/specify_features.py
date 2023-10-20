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
        min_set_for_debug: bool = False,
    ) -> None:
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info
        self.default_fallback = np.nan
        self.default_lookbehind_days = 730

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

    def get_layer_1_specs(self) -> list[PredictorSpec]:
        return [
            PredictorSpec(
                timeseries_df=ldl(),
                feature_base_name="ldl",
                aggregation_fn=mean,
                fallback=np.nan,
                lookbehind_days=self.default_lookbehind_days,
            ),
            PredictorSpec(
                timeseries_df=essential_hypertension(),
                feature_base_name="essential_hypertension",
                aggregation_fn=count,
                fallback=0,
                lookbehind_days=self.default_lookbehind_days,
            ),
        ]

    def get_layer_2_specs(self) -> list[PredictorSpec]:
        specs = PredictorGroupSpec(
            lookbehind_days=[self.default_lookbehind_days],
            named_dataframes=[
                NamedDataframe(
                    df=acute_myocardial_infarction(), name="acute_myocardial_infarction"
                ),
                NamedDataframe(df=pci(), name="pci"),
                NamedDataframe(df=cabg(), name="cabg"),
                NamedDataframe(df=ischemic_stroke(), name="ischemic_stroke"),
                NamedDataframe(df=pad(), name="peripheral_arterial_disease"),
                NamedDataframe(df=hba1c(), name="hba1c"),
                NamedDataframe(df=chronic_lung_disease(), name="chronic_lung_disease"),
            ],
            aggregation_fns=[mean],
            fallback=[np.nan],
        ).create_combinations()

        return specs

    def get_layer_3_specs(self) -> list[PredictorSpec]:
        

    def get_feature_specs(self) -> list[AnySpec]:
        """Get a spec set."""

        if self.min_set_for_debug:
            log.warning(
                "--- !!! Using the minimum set of features for debugging !!! ---",
            )
            return (
                self._get_temporal_predictor_specs()
                + self._get_outcome_specs()
                + self._get_metadata_specs()
            )

        return (
            self._get_temporal_predictor_specs()
            + self._get_static_predictor_specs()
            + self._get_outcome_specs()
            + self._get_metadata_specs()
        )


if __name__ == "__main__":
    main()
