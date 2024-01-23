from psycop.common.feature_generation.application_modules.generate_feature_set import (
    init_wandb_and_generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)
from psycop.projects.t2d.feature_generation.lookbehind_experiment.feature_specification import (
    T2DLookbehindExperimentFeatureSpecifier,
)

if __name__ == "__main__":
    init_wandb_and_generate_feature_set(
        project_info=ProjectInfo(
            project_name="t2d_lookbehind_experiment",
            project_path=OVARTACI_SHARED_DIR / "t2d_lookbehind_experiment",
        ),
        eligible_prediction_times=T2DCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas(),
        feature_specs=T2DLookbehindExperimentFeatureSpecifier().get_feature_specs(),
    )
