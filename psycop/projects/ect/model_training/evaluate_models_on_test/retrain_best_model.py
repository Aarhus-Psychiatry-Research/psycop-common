# Script for re-training best models on spceified splits / congifgs
from pathlib import Path
from typing import Optional

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg


def retrain_best_model(
    experiment_name: str,
    split_outcome: bool = False,
    train_splits: Optional[list[str]] = None,
    test_split: Optional[list[str]] = None,
    trainer: str = "split_trainer_separate_preprocessing",
    data_split_filter: str = "outcomestratified_split_filter",
    test_run_name: str = "evaluated_on_test",
    test_data_path: Optional[list[str]] = None,
):
    if test_data_path is None:
        test_data_path = [
            "E:/shared_resources/ect/feature_set/flattened_datasets/ect_feature_set/ect_feature_set.parquet"
        ]
    if test_split is None:
        test_split = ["val"]
    if train_splits is None:
        train_splits = ["train"]
    test_run_experiment_name = f"{experiment_name}_best_run_{test_run_name}"

    test_run_path = "E:/shared_resources/" + "/ect" + "/eval_runs/" + test_run_experiment_name

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

    best_run_cfg = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=experiment_name, metric="all_oof_BinaryAUROC")
        .get_config()
    )

    preprocessing_pipeline = best_run_cfg.retrieve("trainer.preprocessing_pipeline")

    if split_outcome:
        validation_outcome_col_name = best_run_cfg.retrieve("trainer.validation_outcome_col_name")
        training_outcome_col_name = best_run_cfg.retrieve("trainer.training_outcome_col_name")

        best_run_cfg = (
            best_run_cfg.mut("logger.*.mlflow.experiment_name", test_run_experiment_name)
            .mut("logger.*.disk_logger.run_path", test_run_path)
            .mut("trainer.@trainers", trainer)
            .rem("trainer.validation_outcome_col_name")
            .rem("trainer.training_outcome_col_name")
            .rem("trainer.preprocessing_pipeline")
            .rem("trainer.n_splits")
        )

    else:
        validation_outcome_col_name = best_run_cfg.retrieve("trainer.outcome_col_name")
        training_outcome_col_name = best_run_cfg.retrieve("trainer.outcome_col_name")

        best_run_cfg = (
            best_run_cfg.mut("logger.*.mlflow.experiment_name", test_run_experiment_name)
            .mut("logger.*.disk_logger.run_path", test_run_path)
            .mut("trainer.@trainers", trainer)
            .rem("trainer.outcome_col_name")
            .rem("trainer.preprocessing_pipeline")
            .rem("trainer.n_splits")
        )

    best_run_cfg = (
        best_run_cfg.add("trainer.training_outcome_col_name", training_outcome_col_name)
        .add("trainer.training_preprocessing_pipeline", preprocessing_pipeline)
        .mut(
            "trainer.training_preprocessing_pipeline.*.split_filter.@preprocessing",
            data_split_filter,
        )
        .mut("trainer.training_preprocessing_pipeline.*.split_filter.splits_to_keep", train_splits)
    )

    best_run_cfg = (
        best_run_cfg.add(  # Handle validation dataset
            "trainer.validation_data", best_run_cfg.retrieve("trainer.training_data")
        )
        .add("trainer.validation_data.paths", test_data_path)
        .add("trainer.validation_outcome_col_name", validation_outcome_col_name)
        .add("trainer.validation_preprocessing_pipeline", preprocessing_pipeline)
        .mut(
            "trainer.validation_preprocessing_pipeline.*.split_filter.@preprocessing",
            data_split_filter,
        )
        .mut("trainer.validation_preprocessing_pipeline.*.split_filter.splits_to_keep", test_split)
    )

    train_baseline_model_from_cfg(best_run_cfg)


if __name__ == "__main__":
    retrain_best_model(experiment_name="ECT hparam, structured_only, xgboost, no lookbehind filter", split_outcome=False)
