
from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_registry import BaselineRegistry
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.scz_bp.model_training.populate_scz_bp_registry import populate_scz_bp_registry

# load training data, preprocess with preprocessing pipeline, and apply
# imputation and scaling

# Exp 1: fit model on whole training data, decrease batch size a bit
# Exp 2: fit model on only positive cases
# Sample a large number of synthetic data points

# Save to parquet and concatenate with original data when training in main script


if __name__ == "__main__":
    populate_baseline_registry()
    populate_scz_bp_registry()

    best_experiment = "sczbp/structured_text_xgboost"
    base_config = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=best_experiment, metric="all_oof_BinaryAUROC")
        .get_config()
    )

    # get training data
    training_data_cfg = {"data": base_config["trainer"]["training_data"]}
    training_data = BaselineRegistry.resolve(training_data_cfg)
    # get preprocessing pipeline
    preprocessing_pipeline_cfg = {"preproc": base_config["trainer"]["preprocessing_pipeline"]}
    preprocessing_pipeline = BaselineRegistry.resolve(preprocessing_pipeline_cfg)

    # load and apply the preprocessing pipeline
    training_data_preprocessed = preprocessing_pipeline.apply(data=training_data.load())
