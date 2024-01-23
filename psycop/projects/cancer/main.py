from psycop.common.feature_generation.application_modules.generate_feature_set import (
    init_wandb_and_generate_feature_set,
)
from psycop.projects.cancer.cancer_config import (
    get_cancer_feature_specifications,
    get_cancer_project_info,
)
from psycop.projects.cancer.feature_generation.cohort_definition.cancer_cohort_definer import (
    CancerCohortDefiner,
)
from psycop.projects.cancer.model_training.train_models_in_parallel import train_models_in_parallel

if __name__ == "__main__":
    feature_set_path = init_wandb_and_generate_feature_set(
        project_info=get_cancer_project_info(),
        eligible_prediction_times=CancerCohortDefiner.get_filtered_prediction_times_bundle().prediction_times.frame.to_pandas(),
        feature_specs=get_cancer_feature_specifications(),
        generate_in_chunks=True,
        chunksize=10,
    )

    train_models_in_parallel(dataset_override_path=feature_set_path)
