from joblib import Parallel, delayed
from pathlib import Path
from typing import Optional

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry


def eval_geographic_split_test_set(
    experiment_name: str,
    train_splits: Optional[list[str]] = None,
    test_split: Optional[list[str]] = None,
    test_run_name: str = "evaluated_on_geographic_test",
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

    cfg = (
        MlflowClientWrapper()
        .get_best_run_from_experiment(experiment_name=experiment_name, metric="all_oof_BinaryAUROC")
        .get_config()
    )
    

    outcome_col_name = cfg.retrieve("trainer.outcome_col_name")
    preprocessing_pipeline = cfg.retrieve("trainer.preprocessing_pipeline")

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
            "trainer.training_preprocessing_pipeline.*.split_filter.@preprocessing",
            "regional_data_filter",
        )
        .mut(
            "trainer.training_preprocessing_pipeline.*.split_filter.splits_to_keep",
            train_splits,
        )
        .add("trainer.training_preprocessing_pipeline.*.split_filter.regional_move_df", None)
        .add(
            "trainer.training_preprocessing_pipeline.*.split_filter.timestamp_col_name", "timestamp"
        )
        .add("trainer.training_preprocessing_pipeline.*.split_filter.region_col_name", "region")
        .add(
            "trainer.training_preprocessing_pipeline.*.split_filter.timestamp_cutoff_col_name",
            "first_regional_move_timestamp",
        )
    )

    # Handle validation set setup
    cfg = (
        cfg.add(  # Handle validation dataset
            "trainer.validation_data", cfg.retrieve("trainer.training_data")
        )
        .add("trainer.validation_outcome_col_name", outcome_col_name)
        .add("trainer.validation_preprocessing_pipeline", preprocessing_pipeline)
        .mut(
            "trainer.validation_preprocessing_pipeline.*.split_filter.@preprocessing",
            "regional_data_filter",
        )
        .mut("trainer.validation_preprocessing_pipeline.*.split_filter.splits_to_keep", test_split)
        .add("trainer.validation_preprocessing_pipeline.*.split_filter.regional_move_df", None)
        .add(
            "trainer.validation_preprocessing_pipeline.*.split_filter.timestamp_col_name",
            "timestamp",
        )
        .add("trainer.validation_preprocessing_pipeline.*.split_filter.region_col_name", "region")
        .add(
            "trainer.validation_preprocessing_pipeline.*.split_filter.timestamp_cutoff_col_name",
            "first_regional_move_timestamp",
        )
    )

    train_baseline_model_from_cfg(cfg)


if __name__ == "__main__":
    populate_baseline_registry()

    feature_sets = ["structured_only", "text_only", "structured_text"]

    for feature_set in feature_sets:
        eval_geographic_split_test_set(experiment_name=f"ECT-trunc-and-hp-{feature_set}-xgboost-no-lookbehind-filter",)
