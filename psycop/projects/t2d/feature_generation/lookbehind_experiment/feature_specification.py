import numpy as np
from timeseriesflattener.aggregation_fns import maximum
from timeseriesflattener.feature_specs.group_specs import (
    NamedDataframe,
    OutcomeGroupSpec,
    PredictorGroupSpec,
)
from timeseriesflattener.feature_specs.single_specs import (
    AnySpec,
    OutcomeSpec,
    PredictorSpec,
    StaticSpec,
)

from psycop.common.feature_generation.loaders.raw.load_demographic import sex_female
from psycop.common.feature_generation.loaders.raw.load_lab_results import hba1c
from psycop.projects.t2d.feature_generation.cohort_definition.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)


class T2DLookbehindExperimentFeatureSpecifier:
    def _get_static_predictor_specs(self) -> list[StaticSpec]:
        """Get static predictor specs."""
        return [
            StaticSpec(
                feature_base_name="sex_female",
                timeseries_df=sex_female(),
            ),
        ]

    def _get_outcome_specs(self) -> list[OutcomeSpec]:
        """Get outcome specs."""

        return OutcomeGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=get_first_diabetes_lab_result_above_threshold(),
                    name="first_diabetes_lab_result",
                ),
            ],
            lookahead_days=[year * 365 for year in (1, 2, 3, 4, 5)],
            aggregation_fns=[maximum],
            fallback=[0],
            incident=[True],
        ).create_combinations()

    def _get_temporal_predictor_specs(self) -> list[PredictorSpec]:
        """Get temporal predictor specs."""
        hba1c_within_two_years = [
            PredictorSpec(
                feature_base_name="hba1c_traditional",
                timeseries_df=hba1c(),
                lookbehind_days=730,
                aggregation_fn=maximum,
                fallback=np.nan,
            )
        ]
        hba1c_yearly_interval = PredictorGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=hba1c(),
                    name="hba1c_yearly_interval",
                ),
            ],
            lookbehind_days=[(0, 365), (365, 730)],
            aggregation_fns=[maximum],
            fallback=[np.nan],
        ).create_combinations()

        hba1c_multiple_intervals = PredictorGroupSpec(
            named_dataframes=[
                NamedDataframe(
                    df=hba1c(),
                    name="hba1c_multiple_intervals",
                ),
            ],
            lookbehind_days=[(0, 60), (60, 180), (180, 365), (365, 730)],
            aggregation_fns=[maximum],
            fallback=[np.nan],
        ).create_combinations()

        return hba1c_within_two_years + hba1c_yearly_interval + hba1c_multiple_intervals

    def get_feature_specs(self) -> list[AnySpec]:
        """Get feature specs."""
        return (
            self._get_static_predictor_specs()
            + self._get_temporal_predictor_specs()
            + self._get_outcome_specs()
        )
