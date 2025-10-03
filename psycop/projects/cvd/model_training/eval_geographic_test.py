from pathlib import Path

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry


def eval_geographic_test_set(experiment_name: str, test_run_name: str = "evaluated_on_test"):
    test_run_experiment_name = f"{experiment_name}_best_run_{test_run_name}"
    test_run_path = "E:/shared_resources/" + "/cvd" + "/eval_runs/" + test_run_experiment_name

    if Path(test_run_path).exists():
        while True:
            response = input(
                f"This path '{test_run_path}' already exists. Do you want to potentially overwrite the contents of this folder with new feature sets? (yes/no): "
            )

            if response.lower() not in ["yes", "y", "no", "n"]:
                print("Invalid response. Please enter 'yes/y' or 'no/n'.")
            if response.lower() in ["no", "n"]:
                print("Process stopped.")
                return
            if response.lower() in ["yes", "y"]:
                print("Content of folder may be overwritten.")
                break

    cfg = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=experiment_name, metric="all_oof_BinaryAUROC")
        .get_config()
    )

    outcome_col_name = cfg.retrieve("trainer.outcome_col_name")
    preprocessing_pipeline = cfg.retrieve("trainer.preprocessing_pipeline")

    # Setup for test set
    cfg = (
        cfg.mut("logger.*.mlflow.experiment_name", test_run_experiment_name)
        .mut("logger.*.disk_logger.run_path", test_run_path)
        .mut("trainer.@trainers", "split_trainer_separate_preprocessing")
        .rem("trainer.outcome_col_name")
        .rem("trainer.preprocessing_pipeline")
        .rem("trainer.n_splits")
    )

    # Handle training set setup
    cfg = (
        cfg.add("trainer.training_outcome_col_name", outcome_col_name)
        .add("trainer.training_preprocessing_pipeline", preprocessing_pipeline)
        .mut(
            "trainer.training_preprocessing_pipeline.*.split_filter.splits_to_keep",
            ["train", "val"],
        )
    )

    # Handle validation set setup
    cfg = (
        cfg.add(  # Handle validation dataset
            "trainer.validation_data", cfg.retrieve("trainer.training_data")
        )
        .add("trainer.validation_outcome_col_name", outcome_col_name)
        .add("trainer.validation_preprocessing_pipeline", preprocessing_pipeline)
        .mut("trainer.validation_preprocessing_pipeline.*.split_filter.splits_to_keep", ["test"])
    )

    train_baseline_model_from_cfg(cfg)


if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_cvd_registry()

    eval_geographic_test_set("CVD-hyperparam-tuning-layer-2-xgboost-disk-logged")
