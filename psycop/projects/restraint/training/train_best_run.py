from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import (
    train_baseline_model,
)
from psycop.common.model_training_v2.config.populate_registry import (
    populate_baseline_registry,
)
from psycop.projects.restraint.training.populate_restraint_registry import (
    populate_with_restraint_registry,
)

if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_restraint_registry()
    train_baseline_model(
        MlflowClientWrapper()
        .get_best_run_from_experiment(
            experiment_name="bipolar_test", metric="all_oof_BinaryAUROC"
        )
        .get_config()["project_info"]["experiment_path"]
        / "eval_df.parquet"
    )
