
from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import SczBpCohort
from psycop.projects.scz_bp.scz_bp_config import get_scz_bp_project_info
from psycop.projects.t2d.feature_generation.main import generate_feature_set


if __name__ == "__main__":
    generate_feature_set(
        project_info=get_scz_bp_project_info(),
        eligible_prediction_times=SczBpCohort.get_eligible_prediction_times,
        feature_specs=get_t2d_feature_specifications(),
    )
