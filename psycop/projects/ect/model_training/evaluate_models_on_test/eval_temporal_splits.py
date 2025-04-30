from joblib import Parallel, delayed
from pathlib import Path
from typing import Optional

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry


def eval_stratified_split(
    cfg: PsycopConfig,
    training_end_date: str,
    evaluation_interval: tuple[str, str],
    experiment_name: str,

    test_data_path: Optional[list[str]] = None,
):
    if test_data_path is None:

        test_data_path = [
            "E:/shared_resources/ect/feature_set/flattened_datasets/ect_feature_set/ect_feature_set.parquet"
        ]
    test_run_experiment_name = f"{experiment_name}_best_run_temporal_eval"

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


    
    outcome_col_name: str = cfg.retrieve("trainer.outcome_col_name")

    preprocessing_pipeline = cfg.rem(
        "trainer.preprocessing_pipeline.*.temporal_col_filter"
    ).retrieve("trainer.preprocessing_pipeline")

    # Setup for experiment
    cfg = (
        cfg.mut("logger.*.mlflow.experiment_name", test_run_experiment_name)
        .add(
            "logger.*.mlflow.run_name",
            f"{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}",
        )
        .mut("trainer.@trainers", "split_trainer_separate_preprocessing")
        .rem("trainer.outcome_col_name")  # Separate for train and test later
        .rem("trainer.preprocessing_pipeline")  # Separate for train and test
        .rem("trainer.n_splits")  # Not needed for splits
    )

    # Handle training set setup
    cfg = (
        cfg.add("trainer.training_outcome_col_name", outcome_col_name)
        .add("trainer.training_preprocessing_pipeline", preprocessing_pipeline)
        .add(
            "trainer.training_preprocessing_pipeline.*.date_filter",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": training_end_date,
                "direction": "before",
            },
        )
        .add(
            "trainer.training_preprocessing_pipeline.*.temporal_col_filter",
            {"@preprocessing": "temporal_col_filter"},
        )
        # train and eval across all splits
        .mut(
            "trainer.training_preprocessing_pipeline.*.split_filter.splits_to_keep",
            ["train", "val", "test"],
        )
    )

    # Handle validation set setup
    cfg = (
        cfg.add(  # Handle validation dataset
            "trainer.validation_data", cfg.retrieve("trainer.training_data")
        )
        .add("trainer.validation_outcome_col_name", outcome_col_name)
        .add("trainer.validation_preprocessing_pipeline", preprocessing_pipeline)
        .add(
            "trainer.validation_preprocessing_pipeline.*.date_filter_start",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": evaluation_interval[0],
                "direction": "after-inclusive",
            },
        )
        .add(
            "trainer.validation_preprocessing_pipeline.*.date_filter_end",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": evaluation_interval[1],
                "direction": "before",
            },
        )
        .add(
            "trainer.validation_preprocessing_pipeline.*.temporal_col_filter",
            {"@preprocessing": "temporal_col_filter"},
        )
        # train and eval across all splits
        .mut(
            "trainer.validation_preprocessing_pipeline.*.split_filter.splits_to_keep",
            ["train", "val", "test"],
        )
    )

    return train_baseline_model_from_cfg(cfg)



def evaluate_feature_set_and_year(
    feature_set: str, 
    train_end_year: int,     
) -> tuple[int, dict[int, float | None]]:


    experiment_name = f"ECT-hparam-{feature_set}-xgboost-no-lookbehind-filter"

    evaluation_years = range(train_end_year, 22)
    year_aurocs = {
        y: eval_stratified_split(
            MlflowClientWrapper()
            .get_best_run_from_experiment(
                experiment_name=experiment_name, 
                metric="all_oof_BinaryAUROC",
            )
            .get_config(),
            experiment_name=experiment_name,
            training_end_date=f"20{train_end_year}-01-01",
            evaluation_interval=(f"20{y}-01-01", f"20{y}-12-31"),
        )
        for y in evaluation_years
    }
    return train_end_year, year_aurocs



if __name__ == "__main__":
    populate_baseline_registry()
    feature_sets = ["structured_only", "text_only", "structured_text"]
    train_end_years = range(16, 21)

    # Create a list of all combinations of feature_set and train_end_year
    combinations = [
        (feature_set, train_end_year)
        for feature_set in feature_sets
        for train_end_year in train_end_years
    ]

    # compute in parallel across train end year and feature sets (15 workers)
    # can be flattened to be done across evaluation years as well but, meh
    results = Parallel(n_jobs=len(combinations))(
        delayed(evaluate_feature_set_and_year)(feature_set=feature_set, train_end_year=train_end_year)
        for feature_set, train_end_year in combinations
    )