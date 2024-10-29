"""Main feature generation."""

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set_tsflattener_v1,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.AcuteSomaticAdmission.CohortDefinition.Somatic_admission_cohort_definition import (
    SomaticAdmissionCohortDefiner,
)
from psycop.projects.AcuteSomaticAdmission.FeatureGeneration.modules.specify_features import (
    FeatureSpecifier,
)


def get_somatic_project_info() -> ProjectInfo:
    return ProjectInfo(
        project_name="AcuteSomaticAdmission",
        project_path=OVARTACI_SHARED_DIR / "AcuteSomaticAdmission" / "feature_set",
    )


if __name__ == "__main__":
    project_info = get_somatic_project_info()
    eligible_prediction_times = SomaticAdmissionCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas()
    feature_specs = FeatureSpecifier(
        project_info=project_info,
        min_set_for_debug=False,  # Remember to set to False when generating full dataset
        limited_feature_set=False,
        lookbehind_180d_mean=False,
    ).get_feature_specs()
    generate_feature_set_tsflattener_v1(
        project_info=project_info,
        eligible_prediction_times=eligible_prediction_times,
        feature_specs=feature_specs,
        feature_set_name="somatic_tot_experiments",
    )
