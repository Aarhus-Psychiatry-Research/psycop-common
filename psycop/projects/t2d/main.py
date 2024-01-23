from psycop.common.feature_generation.application_modules.generate_feature_set import (
    init_wandb_and_generate_feature_set,
)
from psycop.projects.t2d.feature_generation.cohort_definition.t2d_cohort_definer import (
    T2DCohortDefiner,
)
from psycop.projects.t2d.model_training.train_models_in_parallel import train_models_in_parallel
from psycop.projects.t2d.t2d_config import get_t2d_feature_specifications, get_t2d_project_info

if __name__ == "__main__":
    feature_set_path = init_wandb_and_generate_feature_set(
        project_info=get_t2d_project_info(),
        eligible_prediction_times=T2DCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas(),
        feature_specs=get_t2d_feature_specifications(),
    )
    train_models_in_parallel(dataset_override_path=feature_set_path)
