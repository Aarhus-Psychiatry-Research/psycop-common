from timeseriesflattener.feature_specs.single_specs import AnySpec

from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.projects.t2d.feature_generation.specify_features import FeatureSpecifier


def get_t2d_project_info() -> ProjectInfo:
    return ProjectInfo(
        project_name="t2d",
        project_path=OVARTACI_SHARED_DIR / "t2d",  # type: ignore
    )


def get_t2d_feature_specifications() -> list[AnySpec]:
    return FeatureSpecifier(
        project_info=get_t2d_project_info(),
        min_set_for_debug=False,  # Remember to set to False when generating full dataset
    ).get_feature_specs()


# For model training configuration, see psycop/projects/t2d/model_training/config/*
