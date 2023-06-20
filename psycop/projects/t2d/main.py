from psycop.projects.t2d.feature_generation.main import generate_feature_set
from psycop.projects.t2d.model_training.train_models_in_parallel import (
    train_models_in_parallel,
)
from psycop.projects.t2d.t2d_config import (
    get_t2d_eligible_prediction_times_as_pandas,
    get_t2d_feature_specifications,
    get_t2d_project_info,
)

if __name__ == "__main__":
    feature_set_path = generate_feature_set(
        project_info=get_t2d_project_info(),
        eligible_prediction_times=get_t2d_eligible_prediction_times_as_pandas(),
        feature_specs=get_t2d_feature_specifications(),
    )
    train_models_in_parallel(dataset_override_path=feature_set_path)
