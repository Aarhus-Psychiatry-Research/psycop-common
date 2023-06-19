from psycop.projects.cancer.feature_generation.main import generate_feature_set
from psycop.projects.cancer.model_training.train_models_in_parallel import (
    train_models_in_parallel,
)

if __name__ == "__main__":
    feature_set_path = generate_feature_set()
    train_models_in_parallel(dataset_override_path=feature_set_path)
