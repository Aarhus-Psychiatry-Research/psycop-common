from pathlib import Path

from joblib import Parallel, delayed

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry


def eval_stratified_split(
    cfg: PsycopConfig, training_end_date: str, evaluation_interval: tuple[str, str], feature_set: str
) -> float:
    outcome_col_name: str = cfg.retrieve("trainer.outcome_col_name")

    # Python dicts are ordered, but we currently remove the timestamp column before generating predictions.
    # Since we cannot insert at a given position in the ordering, we need to remove the timestamp filter,
    # and then re-add it later.
    preprocessing_pipeline = cfg.rem(
        "trainer.preprocessing_pipeline.*.temporal_col_filter"
    ).retrieve("trainer.preprocessing_pipeline")

    # Setup for experiment
    cfg = (
        cfg.mut("logger.*.mlflow.experiment_name", f"ECT, {feature_set} temporal validation")
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
        .mut("trainer.training_preprocessing_pipeline.*.split_filter.splits_to_keep", 
             ["train", "val", "test"])
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
        .mut("trainer.validation_preprocessing_pipeline.*.split_filter.splits_to_keep", 
             ["train", "val", "test"])
    )

    return train_baseline_model_from_cfg(cfg)


def evaluate_feature_set_and_year(feature_set: str, train_end_year: int):
    evaluation_years = range(train_end_year, 22)
    year_aurocs = {
        y: eval_stratified_split(
            MlflowClientWrapper().get_best_run_from_experiment(
                experiment_name=f"ECT hparam, {feature_set}, xgboost, no lookbehind filter", 
                larger_is_better=True, 
                metric="all_oof_BinaryAUROC"
            ).get_config(),
            feature_set=feature_set,
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
combinations = [(feature_set, train_end_year) for feature_set in feature_sets for train_end_year in train_end_years]

# compute in parallel across train end year and feature sets (15 workers)
# can be flattened to be done across evaluation years as well but, meh
results = Parallel(n_jobs=len(combinations))(delayed(evaluate_feature_set_and_year)(feature_set=feature_set, train_end_year=train_end_year) for feature_set, train_end_year in combinations)

