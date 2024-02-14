"""Main feature generation."""


import logging

from timeseriesflattener.aggregation_fns import count, maximum, mean, minimum

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.cvd.feature_generation.cohort_definition.cvd_cohort_definition import (
    CVDCohortDefiner,
)
from psycop.projects.cvd.feature_generation.specify_features import CVDFeatureSpecifier

log = logging.getLogger(__file__)


def get_cvd_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="cvd", project_path=OVARTACI_SHARED_DIR / "cvd" / "feature_set")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y/&m/%d %H:%M:%S",
    )
    log.info("Starting test")

    project_info = get_cvd_project_info()
    eligible_prediction_times = (
        CVDCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas()
    )

    log.info("Getting feature specs")
    feature_specs = CVDFeatureSpecifier().get_feature_specs(
        layer=3,
        aggregation_fns=[
            mean,
            minimum,
            maximum,
            count,
        ],  # [boolean, count, maximum, mean, minimum],
        lookbehind_days=[90, 365, 730],
    )

    log.info("Generating features")
    generate_feature_set(
        project_info=project_info,
        eligible_prediction_times=eligible_prediction_times,
        feature_specs=feature_specs,
        feature_set_name="cvd_lookbehind_experiments",
    )
