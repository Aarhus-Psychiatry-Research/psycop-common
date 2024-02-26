"""Main feature generation."""


from timeseriesflattener.v1.aggregation_fns import mean

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    CVDCohortDefiner,
)
from psycop.projects.cvd.feature_generation.specify_features import CVDFeatureSpecifier


def get_cvd_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="cvd", project_path=OVARTACI_SHARED_DIR / "cvd" / "feature_set")


if __name__ == "__main__":
    project_info = get_cvd_project_info()
    eligible_prediction_times = (
        CVDCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas()
    )
    feature_specs = CVDFeatureSpecifier().get_feature_specs(
        layer=1,
        aggregation_fns=[mean],  # [boolean, count, maximum, mean, minimum],
        lookbehind_days=[90, 365, 730],
    )

    generate_feature_set(
        project_info=project_info,
        eligible_prediction_times=eligible_prediction_times,
        feature_specs=feature_specs,
        feature_set_name="cvd_lookbehind_experiments",
    )
