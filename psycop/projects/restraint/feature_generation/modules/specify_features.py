"""Feature specification module."""
import logging
from typing import Callable, Union

import numpy as np
from timeseriesflattener.aggregation_fns import (
    boolean,
    change_per_day,
    count,
    latest,
    maximum,
    mean,
    minimum,
    summed,
    variance,
)
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    PredictorSpec,
    StaticSpec,
)

from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.feature_generation.loaders.raw.load_coercion import (
    af_helbredsmaessige_grunde,
    af_legemlig_lidelse,
    baelte,
    beroligende_medicin,
    ect,
    farlighed,
    fastholden,
    medicinering,
    paa_grund_af_farlighed,
    remme,
    skema_1,
    skema_2_without_nutrition,
    skema_3,
    tvangsindlaeggelse,
    tvangstilbageholdelse,
)
from psycop.common.feature_generation.loaders.raw.load_demographic import (
    sex_female,
)
from psycop.common.feature_generation.loaders.raw.load_diagnoses import (
    depressive_disorders,
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
    manic_and_bipolar,
)
from psycop.common.feature_generation.loaders.raw.load_medications import (
    alcohol_abstinence,
    antidepressives,
    antipsychotics,
    anxiolytics,
    clozapine,
    hypnotics,
    hypnotics_and_rivotril,
    lithium,
    mood_stabilisers,
    nervous_system_stimulants,
    non_sedative_antipsychotics,
    opioid_dependence,
    pregabaline,
    sedative_antipsychotics,
)
from psycop.common.feature_generation.loaders.raw.load_structured_sfi import (
    bmi,
    broeset_violence_checklist,
    hamilton_d17,
    height_in_cm,
    mas_m,
    no_temporary_leave,
    selvmordsrisiko,
    supervised_temporary_leave,
    temporary_leave,
    unsupervised_temporary_leave,
    weight_in_kg,
)
from psycop.common.feature_generation.loaders.raw.load_visits import (
    admissions,
    ambulatory_visits,
    emergency_visits,
    physical_visits,
    physical_visits_to_psychiatry,
    physical_visits_to_somatic,
)

log = logging.getLogger(__name__)


class FeatureSpecifier:
    """Specify features based on prediction time."""

    def __init__(self, project_info: ProjectInfo, min_set_for_debug: bool = False):
        self.min_set_for_debug = min_set_for_debug
        self.project_info = project_info

    def _get_static_predictor_specs(self) -> list[StaticSpec]:
        """Get static predictor specs."""
        return [
            StaticSpec(
                timeseries_df=sex_female(),
                feature_base_name="sex_female",
                prefix=self.project_info.prefix.predictor,
            ),
        ]

    def _get_weight_and_height_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get weight and height specs."""
        log.info("-------- Generating weight and height specs --------")

        weight_height_bmi = PredictorGroupSpec(
            named_dataframes=[
                NamedDataframe(df=weight_in_kg(), name="weight_in_kg"),
                NamedDataframe(df=height_in_cm(), name="height_in_cm"),
                NamedDataframe(df=bmi(), name="bmi"),
            ],
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[np.nan],
            prefix=self.project_info.prefix.predictor,
        ).create_combinations()

        return weight_height_bmi

    def _get_visits_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get visits specs."""
        log.info("-------- Generating visits specs --------")

        visits = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=physical_visits(), name="physical_visits"),
                NamedDataframe(
                    df=physical_visits_to_psychiatry(),
                    name="physical_visits_to_psychiatry",
                ),
                NamedDataframe(
                    df=physical_visits_to_somatic(),
                    name="physical_visits_to_somatic",
                ),
                NamedDataframe(
                    df=admissions(),
                    name="admissions",
                ),
                NamedDataframe(
                    df=ambulatory_visits(),
                    name="ambulatory_visits",
                ),
                NamedDataframe(
                    df=emergency_visits(),
                    name="emergency_visits",
                ),
                NamedDataframe(
                    df=admissions(shak_code=6600, shak_sql_operator="="),
                    name="admissions_to_psychiatry",
                ),
                NamedDataframe(
                    df=admissions(shak_code=6600, shak_sql_operator="!="),
                    name="admissions_to_somatic",
                ),
                NamedDataframe(
                    df=ambulatory_visits(shak_code=6600, shak_sql_operator="="),
                    name="ambulatory_visits_to_psychiatry",
                ),
                NamedDataframe(
                    df=ambulatory_visits(shak_code=6600, shak_sql_operator="!="),
                    name="ambulatory_visits_to_somatic",
                ),
                NamedDataframe(
                    df=emergency_visits(shak_code=6600, shak_sql_operator="="),
                    name="emergency_visits_to_psychiatry",
                ),
                NamedDataframe(
                    df=emergency_visits(shak_code=6600, shak_sql_operator="!="),
                    name="emergency_visits_to_somatic",
                ),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return visits

    def _get_diagnoses_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get diagnoses specs."""
        log.info("-------- Generating diagnoses specs --------")

        psychiatric_diagnoses = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=f0_disorders(), name="f0_disorders"),
                NamedDataframe(df=f1_disorders(), name="f1_disorders"),
                NamedDataframe(df=f2_disorders(), name="f2_disorders"),
                NamedDataframe(df=f3_disorders(), name="f3_disorders"),
                NamedDataframe(df=manic_and_bipolar(), name="manic_and_bipolar"),
                NamedDataframe(df=depressive_disorders(), name="depressive_disorders"),
                NamedDataframe(df=f4_disorders(), name="f4_disorders"),
                NamedDataframe(df=f5_disorders(), name="f5_disorders"),
                NamedDataframe(df=f6_disorders(), name="f6_disorders"),
                NamedDataframe(df=f7_disorders(), name="f7_disorders"),
                NamedDataframe(df=f8_disorders(), name="f8_disorders"),
                NamedDataframe(df=f9_disorders(), name="f9_disorders"),
            ),
            aggregation_fns=resolve_multiple,
            lookbehind_days=interval_days,
            fallback=[0],
        ).create_combinations()

        return psychiatric_diagnoses

    def _get_medication_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get medication specs."""
        log.info("-------- Generating medication specs --------")

        psychiatric_medications = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=antipsychotics(), name="antipsychotics"),
                NamedDataframe(
                    df=sedative_antipsychotics(),
                    name="sedative_antipsychotics",
                ),
                NamedDataframe(
                    df=non_sedative_antipsychotics(),
                    name="non_sedative_antipsychotics",
                ),
                NamedDataframe(
                    df=antipsychotics(administration_method="Fast"),
                    name="antipsychotics_fast",
                ),
                NamedDataframe(
                    df=antipsychotics(administration_method="PN"),
                    name="antipsychotics_pn",
                ),
                NamedDataframe(
                    df=antipsychotics(administration_method="Engangs"),
                    name="antipsychotics_engangs",
                ),
                NamedDataframe(
                    df=antipsychotics(administration_route="IM"),
                    name="antipsychotics_im",
                ),
                NamedDataframe(
                    df=antipsychotics(administration_route="PO"),
                    name="antipsychotics_po",
                ),
                NamedDataframe(
                    df=sedative_antipsychotics(administration_method="Fast"),
                    name="sedative_antipsychotics_fast",
                ),
                NamedDataframe(
                    df=sedative_antipsychotics(administration_method="PN"),
                    name="sedative_antipsychotics_pn",
                ),
                NamedDataframe(
                    df=sedative_antipsychotics(administration_method="Engangs"),
                    name="sedative_antipsychotics_engangs",
                ),
                NamedDataframe(
                    df=sedative_antipsychotics(administration_route="IM"),
                    name="sedative_antipsychotics_im",
                ),
                NamedDataframe(
                    df=sedative_antipsychotics(administration_route="PO"),
                    name="sedative_antipsychotics_po",
                ),
                NamedDataframe(
                    df=non_sedative_antipsychotics(administration_method="Fast"),
                    name="non_sedative_antipsychotics_fast",
                ),
                NamedDataframe(
                    df=non_sedative_antipsychotics(administration_method="PN"),
                    name="non_sedative_antipsychotics_pn",
                ),
                NamedDataframe(
                    df=non_sedative_antipsychotics(administration_method="Engangs"),
                    name="non_sedative_antipsychotics_engangs",
                ),
                NamedDataframe(
                    df=non_sedative_antipsychotics(administration_route="IM"),
                    name="non_sedative_antipsychotics_im",
                ),
                NamedDataframe(
                    df=non_sedative_antipsychotics(administration_route="PO"),
                    name="non_sedative_antipsychotics_po",
                ),
                NamedDataframe(
                    df=clozapine(administration_method="Fast"),
                    name="clozapine_fast",
                ),
                NamedDataframe(
                    df=clozapine(administration_method="PN"),
                    name="clozapine_pn",
                ),
                NamedDataframe(
                    df=clozapine(administration_method="Engangs"),
                    name="clozapine_engangs",
                ),
                NamedDataframe(
                    df=clozapine(administration_route="IM"),
                    name="clozapine_im",
                ),
                NamedDataframe(
                    df=clozapine(administration_route="PO"),
                    name="clozapine_po",
                ),
                NamedDataframe(df=anxiolytics(), name="anxiolytics"),
                NamedDataframe(
                    df=anxiolytics(administration_method="Fast"),
                    name="anxiolytics_fast",
                ),
                NamedDataframe(
                    df=anxiolytics(administration_method="PN"),
                    name="anxiolytics_pn",
                ),
                NamedDataframe(
                    df=anxiolytics(administration_method="Engangs"),
                    name="anxiolytics_engangs",
                ),
                NamedDataframe(
                    df=pregabaline(administration_method="Fast"),
                    name="pregabaline_fast",
                ),
                NamedDataframe(
                    df=pregabaline(administration_method="PN"),
                    name="pregabaline_pn",
                ),
                NamedDataframe(
                    df=pregabaline(administration_method="Engangs"),
                    name="pregabaline_engangs",
                ),
                NamedDataframe(df=hypnotics(), name="hypnotics_and_sedatives"),
                NamedDataframe(
                    df=hypnotics(administration_method="Fast"),
                    name="hypnotics_and_sedatives_fast",
                ),
                NamedDataframe(
                    df=hypnotics(administration_method="PN"),
                    name="hypnotics_and_sedatives_pn",
                ),
                NamedDataframe(
                    df=hypnotics(administration_method="Engangs"),
                    name="hypnotics_and_sedatives_engangs",
                ),
                NamedDataframe(
                    df=hypnotics_and_rivotril(),
                    name="hypnotics_and_sedatives_with_rivotril",
                ),
                NamedDataframe(
                    df=hypnotics_and_rivotril(administration_method="Fast"),
                    name="hypnotics_and_sedatives_with_rivotril_fast",
                ),
                NamedDataframe(
                    df=hypnotics_and_rivotril(administration_method="PN"),
                    name="hypnotics_and_sedatives_with_rivotril_pn",
                ),
                NamedDataframe(
                    df=hypnotics_and_rivotril(administration_method="Engangs"),
                    name="hypnotics_and_sedatives_with_rivotril_engangs",
                ),
                NamedDataframe(df=antidepressives(), name="antidepressives"),
                NamedDataframe(df=lithium(), name="lithium"),
                NamedDataframe(df=alcohol_abstinence(), name="alcohol_abstinence"),
                NamedDataframe(df=opioid_dependence(), name="opioid_dependence"),
                NamedDataframe(df=clozapine(), name="clozapine"),
                NamedDataframe(df=pregabaline(), name="pregabaline"),
                NamedDataframe(df=mood_stabilisers(), name="mood_stabilisers"),
                NamedDataframe(
                    df=nervous_system_stimulants(),
                    name="nervous_system_stimulants",
                ),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return psychiatric_medications

    def _get_schema_1_and_2_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get schema 1 and schema 2 coercion specs."""
        log.info("-------- Generating schema 1 and schema 2 coercion specs --------")

        coercion = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=skema_1(), name="skema_1"),
                NamedDataframe(df=tvangsindlaeggelse(), name="tvangsindlaeggelse"),
                NamedDataframe(
                    df=tvangstilbageholdelse(),
                    name="tvangstilbageholdelse",
                ),
                NamedDataframe(
                    df=paa_grund_af_farlighed(),
                    name="paa_grund_af_farlighed",
                ),  # røde papirer
                NamedDataframe(
                    df=af_helbredsmaessige_grunde(),
                    name="af_helbredsmaessige_grunde",
                ),  # gule papirer
                NamedDataframe(
                    df=skema_2_without_nutrition(),
                    name="skema_2_without_nutrition",
                ),
                NamedDataframe(df=medicinering(), name="medicinering"),
                NamedDataframe(df=ect(), name="ect"),
                NamedDataframe(df=af_legemlig_lidelse(), name="af_legemlig_lidelse"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return coercion

    def _get_schema_1_and_2_current_status_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get "current" status of schema 1 and schema 2 coercion."""
        log.info(
            "Generating schema 1 and schema 2 current status specs --------",
        )

        coercion = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=skema_1(unpack_to_intervals=True), name="skema_1"),
                NamedDataframe(
                    df=tvangsindlaeggelse(unpack_to_intervals=True),
                    name="tvangsindlaeggelse",
                ),
                NamedDataframe(
                    df=tvangstilbageholdelse(unpack_to_intervals=True),
                    name="tvangstilbageholdelse",
                ),
                NamedDataframe(
                    df=paa_grund_af_farlighed(unpack_to_intervals=True),
                    name="paa_grund_af_farlighed",
                ),  # røde papirer
                NamedDataframe(
                    df=af_helbredsmaessige_grunde(unpack_to_intervals=True),
                    name="af_helbredsmaessige_grunde",
                ),  # gule papirer
                NamedDataframe(
                    df=skema_2_without_nutrition(unpack_to_intervals=True),
                    name="skema_2_without_nutrition",
                ),
                NamedDataframe(
                    df=medicinering(unpack_to_intervals=True),
                    name="medicinering",
                ),
                NamedDataframe(df=ect(unpack_to_intervals=True), name="ect"),
                NamedDataframe(
                    df=af_legemlig_lidelse(unpack_to_intervals=True),
                    name="af_legemlig_lidelse",
                ),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return coercion

    def _get_schema_3_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get schema 3 coercion specs."""
        log.info("-------- Generating schema 3 coercion specs --------")

        coercion = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=skema_3(), name="skema_3"),
                NamedDataframe(df=fastholden(), name="fastholden"),
                NamedDataframe(df=baelte(), name="baelte"),
                NamedDataframe(df=remme(), name="remme"),
                NamedDataframe(df=farlighed(), name="farlighed"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return coercion

    def _get_forced_medication_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get forced medication coercion specs."""
        log.info("-------- Generating forced medication coercion specs --------")

        beroligende_medicin_df = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=beroligende_medicin(), name="beroligende_medicin"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return beroligende_medicin_df

    def _get_structured_sfi_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get structured sfi specs."""
        log.info("-------- Generating structured sfi specs --------")

        structured_sfi = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(
                    df=broeset_violence_checklist(),
                    name="broeset_violence_checklist",
                ),
                NamedDataframe(df=selvmordsrisiko(), name="selvmordsrisiko"),
                NamedDataframe(df=hamilton_d17(), name="hamilton_d17"),
                NamedDataframe(df=mas_m(), name="mas_m"),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[np.nan],
        ).create_combinations()

        return structured_sfi

    def _get_temporary_leave_specs(
        self,
        resolve_multiple: list[Callable],
        interval_days: list[float],
    ) -> list[PredictorSpec]:
        """Get structured sfi specs."""
        log.info("-------- Generating temporary leave specs --------")

        temporary_leave_specs = PredictorGroupSpec(
            named_dataframes=(
                NamedDataframe(df=no_temporary_leave(), name="no_temporary_leave"),
                NamedDataframe(df=temporary_leave(), name="temporary_leave"),
                NamedDataframe(
                    df=supervised_temporary_leave(),
                    name="supervised_temporary_leave",
                ),
                NamedDataframe(
                    df=unsupervised_temporary_leave(),
                    name="unsupervised_temporary_leave",
                ),
            ),
            lookbehind_days=interval_days,
            aggregation_fns=resolve_multiple,
            fallback=[0],
        ).create_combinations()

        return temporary_leave_specs

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Generate predictor spec list."""
        log.info("-------- Generating temporal predictor specs --------")

        if self.min_set_for_debug:
            return [
                PredictorSpec(
                    feature_base_name="f0_disorders",
                    timeseries_df=f0_disorders(),
                    lookbehind_days=100,
                    aggregation_fn=boolean,
                    fallback=np.nan,
                    prefix=self.project_info.prefix.predictor,
                ),
            ]

        resolve_multiple = [boolean, count]
        interval_days = [10.0, 30.0, 180.0, 365.0, 730.0]

        latest_weight_height_bmi = self._get_weight_and_height_specs(
            resolve_multiple=[latest],
            interval_days=interval_days,
        )

        visits = self._get_visits_specs(
            resolve_multiple=[*resolve_multiple, summed],
            interval_days=interval_days,
        )

        diagnoses = self._get_diagnoses_specs(
            resolve_multiple=[boolean],
            interval_days=interval_days,
        )

        medications = self._get_medication_specs(
            resolve_multiple=resolve_multiple,
            interval_days=[1, 3, 7, *interval_days],
        )

        schema_1_schema_2_coercion = self._get_schema_1_and_2_specs(
            resolve_multiple=[*resolve_multiple, summed],
            interval_days=[7, *interval_days],
        )

        schema_1_schema_2_coercion_current_status = (
            self._get_schema_1_and_2_current_status_specs(
                resolve_multiple=[boolean],
                interval_days=[1, 3],
            )
        )

        schema_3_coercion = self._get_schema_3_specs(
            resolve_multiple=[*resolve_multiple, summed],
            interval_days=[730],
        )

        forced_medication_coercion = self._get_forced_medication_specs(
            resolve_multiple=resolve_multiple,
            interval_days=[730],
        )

        structured_sfi = self._get_structured_sfi_specs(
            resolve_multiple=[mean, maximum, minimum, change_per_day, variance],
            interval_days=[1, 3, 7, *interval_days],
        )

        temporary_leave = self._get_temporary_leave_specs(
            resolve_multiple=resolve_multiple,
            interval_days=[1, 3, 7, *interval_days],
        )

        return (
            latest_weight_height_bmi
            + visits
            + medications
            + diagnoses
            + schema_1_schema_2_coercion
            + schema_1_schema_2_coercion_current_status
            + schema_3_coercion
            + forced_medication_coercion
            + structured_sfi
            + temporary_leave
        )

    def get_feature_specs(
        self,
    ) -> list[Union[StaticSpec, PredictorSpec]]:
        """Get a spec set."""

        if self.min_set_for_debug:
            return self._get_diagnoses_specs(resolve_multiple=[boolean], interval_days=[30])  # type: ignore

        return self._get_static_predictor_specs() + self._get_temporal_predictor_specs()
