import time

from psycop.common.feature_generation.application_modules.generate_feature_set import (
    generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.bipolar.cohort_definition.bipolar_cohort_definition import BipolarCohortDefiner
from psycop.projects.bipolar.feature_generation.bp_specify_features import BpFeatureSpecifier


def get_bp_project_info() -> ProjectInfo:
    return ProjectInfo(project_name="bipolar", project_path=OVARTACI_SHARED_DIR / "bipolar")


if __name__ == "__main__":
    t0 = time.time()
    generate_feature_set(
        project_info=get_bp_project_info(),
        eligible_prediction_times_frame=BipolarCohortDefiner.get_prediction_times(
            interval_days=100
        ),
        feature_specs=BpFeatureSpecifier().get_feature_specs(max_layer=5, lookbehind_days=[200]),
        n_workers=10,
        do_dataset_description=False,
        feature_set_name="structured_predictors_5_layer_interval_days_100",
    )
    t = time.time()
    print(f"Time taken: {t - t0}")
