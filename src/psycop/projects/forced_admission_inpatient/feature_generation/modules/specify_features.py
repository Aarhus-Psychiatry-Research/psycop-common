"""Feature specification module."""
import logging
from typing import Union

import numpy as np
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from timeseriesflattener.feature_spec_objects import (
    BaseModel,
    OutcomeSpec,
    PredictorGroupSpec,
    PredictorSpec,
    StaticSpec,
    _AnySpec,
)

log = logging.getLogger(__name__)


class SpecSet(BaseModel):
    """A set of unresolved specs, ready for resolving."""

    temporal_predictors: list[PredictorSpec]
    static_predictors: list[StaticSpec]
    outcomes: list[OutcomeSpec]
    metadata: list[_AnySpec]


class FeatureSpecifier:
    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_static_predictor_specs(self) -> list[StaticSpec]:
        """Get static predictor specs."""
        return [
            StaticSpec(
                values_loader="sex_female", #type: ignore
                input_col_name_override="sex_female",
                prefix=self.project_info.prefix.predictor,
                feature_name="sex_female",
            ),
        ]

    def _get_visits_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get visits specs."""
        log.info("-------- Generating visits specs --------")

        visits = PredictorGroupSpec(
            values_loader=(
                "physical_visits_to_psychiatry",
                "physical_visits_to_somatic",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
            feature_name="physical_visits",
        ).create_combinations()

        return visits

    def _get_admissions_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get admissions specs."""
        log.info("-------- Generating admissions specs --------")

        admissions = PredictorGroupSpec(
            values_loader=("admissions",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
            feature_name="admissions",
        ).create_combinations()

        return admissions

    def _get_medication_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get medication specs."""
        log.info("-------- Generating medication specs --------")

        psychiatric_medications = PredictorGroupSpec(
            values_loader=(
                "antipsychotics",
                "first_gen_antipsychotics",
                "second_gen_antipsychotics",
                "olanzapine",
                "clozapine",
                "anxiolytics",
                "hypnotics and sedatives",
                "antidepressives",
                "lithium",
                "hyperactive disorders medications",
                "alcohol_abstinence",
                "opioid_dependence",
                "analgesics",
                "olanzapine_depot",
                "aripiprazole_depot",
                "risperidone_depot",
                "paliperidone_depot",
                "haloperidol_depot",
                "perphenazine_depot",
                "zuclopenthixol_depot",
            ),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
            feature_name="phychiatric_medications",
        ).create_combinations()

        return psychiatric_medications

    def _get_diagnoses_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get diagnoses specs."""
        log.info("-------- Generating diagnoses specs --------")

        psychiatric_diagnoses = PredictorGroupSpec(
            values_loader=(
                "f0_disorders",
                "f1_disorders",
                "f2_disorders",
                "schizophrenia",
                "schizoaffective",
                "f3_disorders",
                "manic_and_bipolar",
                "f4_disorders",
                "f5_disorders",
                "f6_disorders",
                "cluster_b",
                "f7_disorders",
                "f8_disorders",
                "f9_disorders",
            ),
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
            feature_name="psychiatric_diagnoses",
        ).create_combinations()

        return psychiatric_diagnoses

    def _get_coercion_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get coercion specs."""
        log.info("-------- Generating coercion specs --------")

        coercion = PredictorGroupSpec(
            values_loader=(
                "skema_1",
                "tvangstilbageholdelse",
                "skema_2_without_nutrition",
                "medicinering",
                "ect",
                "af_legemlig_lidelse",
                "skema_3",
                "fastholden",
                "baelte",
                "remme",
                "farlighed",
            ),
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            lookbehind_days=interval_days,
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
            feature_name="coercion_types",
        ).create_combinations()

        return coercion

    def _get_beroligende_medicin_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get beroligende medicin specs."""
        log.info("-------- Generating beroligende medicicn specs --------")

        beroligende_medicin = PredictorGroupSpec(
            values_loader=("beroligende_medicin",),
            lookbehind_days=interval_days,
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            fallback=[0],
            allowed_nan_value_prop=allowed_nan_value_prop,
            feature_name="beroligende_medicin",
        ).create_combinations()

        return beroligende_medicin

    def _get_structured_sfi_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get structured sfi specs."""
        log.info("-------- Generating structured sfi specs --------")

        structured_sfi = PredictorGroupSpec(
            values_loader=(
                "broeset_violence_checklist",
                "selvmordsrisiko",
                "hamilton_d17",
                "mas_m",
            ),
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            lookbehind_days=interval_days,
            fallback=[np.nan],
            allowed_nan_value_prop=allowed_nan_value_prop,
            feature_name="structured_sfi",
        ).create_combinations()

        return structured_sfi

    def _get_lab_result_specs(
        self,
        resolve_multiple: list[str],
        interval_days: list[float],
        allowed_nan_value_prop: list[float],
    ) -> list[PredictorSpec]:
        """Get lab result specs."""
        log.info("-------- Generating lab result specs --------")

        lab_results = PredictorGroupSpec(
            values_loader=(
                "p_lithium",
                "p_clozapine",
                "p_olanzapine",
                "p_aripiprazol",
                "p_risperidone",
                "p_paliperidone",
                "p_haloperidol",
                "p_paracetamol",
                "p_ethanol",
                "p_nortriptyline",
                "p_clomipramine",
                "cancelled_standard_lab_results",
            ),
            resolve_multiple_fn=resolve_multiple,  # type: ignore
            lookbehind_days=interval_days,
            fallback=[np.nan],
            allowed_nan_value_prop=allowed_nan_value_prop,
            feature_name="lab_results",
        ).create_combinations()

        return lab_results

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("-------- Generating temporal predictor specs --------")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    values_loader="schizophrenia",
                    lookbehind_days=30,
                    resolve_multiple_fn="max",
                    fallback=np.nan,
                    allowed_nan_value_prop=0,
                    prefix=self.project_info.prefix.predictor,
                ),
            ]

        interval_days = [10.0, 30.0, 180.0, 365.0]
        allowed_nan_value_prop = [0.0]

        visits = self._get_visits_specs(
            resolve_multiple=["count"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        admissions = self._get_admissions_specs(
            resolve_multiple=["count", "sum"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=["count", "bool"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        medications = self._get_medication_specs(
            resolve_multiple=["count", "bool"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        beroligende_medicin = self._get_beroligende_medicin_specs(
            resolve_multiple=["count", "bool"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        coercion = self._get_coercion_specs(
            resolve_multiple=["count", "sum", "bool"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        structured_sfi = self._get_structured_sfi_specs(
            resolve_multiple=["mean", "max", "min", "change_per_day", "variance"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        lab_results = self._get_lab_result_specs(
            resolve_multiple=["max", "min", "mean", "latest"],
            interval_days=interval_days,
            allowed_nan_value_prop=allowed_nan_value_prop,
        )

        return (
            visits
            + admissions
            + medications
            + diagnoses
            + beroligende_medicin
            + coercion
            + structured_sfi
            + lab_results
        )

    def get_feature_specs(self) -> list[Union[StaticSpec, PredictorSpec]]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return (self._get_temporal_predictor_specs()
                    +self. _get_static_predictor_specs()
            )

        return self._get_temporal_predictor_specs() + self._get_static_predictor_specs()
