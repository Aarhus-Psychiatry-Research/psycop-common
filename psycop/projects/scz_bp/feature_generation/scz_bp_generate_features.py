from psycop.common.feature_generation.application_modules.generate_feature_set import (
    init_wandb_and_generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.feature_generation.scz_bp_specify_features import (
    SczBpFeatureSpecifier,
)


def get_scz_bp_project_info() -> ProjectInfo:
    return ProjectInfo(
        project_name="scz_bp",
        project_path=OVARTACI_SHARED_DIR / "scz_bp" / "initial_feature_set",
    )


if __name__ == "__main__":
    init_wandb_and_generate_feature_set(
        project_info=get_scz_bp_project_info(),
        eligible_prediction_times=SczBpCohort.get_filtered_prediction_times_bundle().prediction_times.to_pandas(),
        feature_specs=SczBpFeatureSpecifier().get_feature_specs(
            max_layer=7,
            lookbehind_days=[730],
        ),
        generate_in_chunks=True,
        feature_set_name="layer7",
    )
