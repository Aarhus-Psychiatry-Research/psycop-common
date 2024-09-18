from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry


def eval_stratified_split(
    cfg: PsycopConfig, training_end_date: str, evaluation_interval: tuple[str, str]
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
        cfg.mut("logger.*.mlflow.experiment_name", "Inpatient v2, temporal validation")
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
    )

    return train_baseline_model_from_cfg(cfg)


if __name__ == "__main__":
    populate_baseline_registry()

    endyear2aurocs = {}
    for train_end_year in range(16, 21):
        evaluation_years = range(train_end_year, 22)
        year_aurocs = {
            y: eval_stratified_split(
                PsycopConfig().from_disk(Path(__file__).parent / "inpatient_v2.cfg"),
                training_end_date=f"20{train_end_year}-01-01",
                evaluation_interval=(f"20{y}-01-01", f"20{y}-12-31"),
            )
            for y in evaluation_years
        }
        endyear2aurocs[train_end_year] = year_aurocs

    print(endyear2aurocs)
