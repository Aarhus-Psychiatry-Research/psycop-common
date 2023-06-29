from timeseriesflattener.feature_specs.single_specs import AnySpec

from psycop.common.feature_generation.application_modules.project_setup import (
    ProjectInfo,
)
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.scz_bp.feature_generation.specify_features import SczBpFeatureSpecifier


def get_scz_bp_project_info() -> ProjectInfo:
    return ProjectInfo(
        project_name="scz_bp",
        project_path=OVARTACI_SHARED_DIR / "scz_bp",
    )

def get_scz_bp_feature_specifications() -> list[AnySpec]:
    return SczBpFeatureSpecifier(
        project_info=get_scz_bp_project_info(),
        min_set_for_debug=True,  # Remember to set to False when generating full dataset
    ).get_feature_specs()


# For model training configuration, see psycop/projects/scz_bp/model_training/config/*

if __name__ == "__main__":

    spesc = get_scz_bp_feature_specifications()