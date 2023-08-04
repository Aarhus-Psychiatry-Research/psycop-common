from psycop.common.feature_generation.application_modules.generate_feature_set import (
    init_wandb_and_generate_feature_set,
)
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)
from psycop.projects.scz_bp.scz_bp_config import (
    get_scz_bp_feature_specifications,
    get_scz_bp_project_info,
)

if __name__ == "__main__":
    init_wandb_and_generate_feature_set(
        project_info=get_scz_bp_project_info(),
        eligible_prediction_times=SczBpCohort.get_filtered_prediction_times_bundle().prediction_times.to_pandas(),
        feature_specs=get_scz_bp_feature_specifications(),
        generate_in_chunks=True,
        chunksize=200,
    )
