from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.t2d_extended.model_training.populate_t2d_registry import populate_with_t2d_extended_registry


def eval_temporal_stability(
    cfg: PsycopConfig, training_start_date: str, training_end_date: str, evaluation_interval: tuple[str, str]
) -> float:
    outcome_col_name: str = cfg.retrieve("trainer.outcome_col_name")
    preprocessing_pipeline = cfg.rem(
        "trainer.preprocessing_pipeline.*.temporal_col_filter"
    ).retrieve("trainer.preprocessing_pipeline")

    # Setup for experiment
    cfg = (
        cfg.mut("logger.*.mlflow.experiment_name", "T2D-extended, temporal validation")
        .mut("logger.*.disk.run_path", f"E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/{training_start_date}_{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}") # fh: TODO change "E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation" to automatic exp path                 # .mut("logger.*.disk.run_path", f"E:/shared_resources/t2d_extended/training/T2D-extended, temporal validation/{training_start_date}_{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}")
        .add(
            "logger.*.mlflow.run_name",
            f"{training_start_date}_{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}",
        )
        .mut("trainer.@trainers", "split_trainer_separate_preprocessing")
        .rem("trainer.outcome_col_name")
        .rem("trainer.preprocessing_pipeline")
        .rem("trainer.n_splits")
    )

    # Python dicts are ordered, but we remove the timestamp column before generating predictions.
    # To filter based on timestamp, we need to add the filter before temporal columns are removed.
    # These shenanigans are needed to insert.
    # Handle training set setup
    cfg = (
        cfg.add("trainer.training_outcome_col_name", outcome_col_name)
        .add("trainer.training_preprocessing_pipeline", preprocessing_pipeline)
        .add(
            "trainer.training_preprocessing_pipeline.*.date_filter_start",
            {
                "@preprocessing": "date_filter",
                "column_name": "timestamp",
                "threshold_date": training_start_date,
                "direction": "after-inclusive",
            },
        )
        .add(
            "trainer.training_preprocessing_pipeline.*.date_filter_end",
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
    populate_with_t2d_extended_registry()

    aurocs = {
        y: eval_temporal_stability(
            PsycopConfig().from_disk(Path(__file__).parent / "t2d_extended.cfg"),
            training_start_date="2013-01-01",
            training_end_date="2018-01-01",
            evaluation_interval=(f"20{y}-01-01", f"20{y}-12-31"),
        )
        for y in range(18, 24)
    }


    print(aurocs)
