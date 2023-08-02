from psycop.common.feature_generation.application_modules.generate_feature_set import (
    init_wandb_and_generate_feature_set,
)
from psycop.common.feature_generation.application_modules.project_setup import ProjectInfo
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.scz_bp_config import (
    get_scz_bp_feature_specifications,
    get_scz_bp_project_info,
)
import pandas as pd
from timeseriesflattener.feature_specs.single_specs import AnySpec


def generate_feature_set_in_chunks(project_info: ProjectInfo, eligible_prediction_times: pd.DataFrame, feature_specs: list[AnySpec]) -> None:



if __name__ == "__main__":
    init_wandb_and_generate_feature_set(
        project_info=get_scz_bp_project_info(),
        eligible_prediction_times=SczBpCohort.get_filtered_prediction_times_bundle().prediction_times.to_pandas(),
        feature_specs=get_scz_bp_feature_specifications()[:500],
    )
