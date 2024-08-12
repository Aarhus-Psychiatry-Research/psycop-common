from pathlib import Path

from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.cvd.model_training.populate_cvd_registry import populate_with_cvd_registry


def test_set(cfg: PsycopConfig):
    outcome_col_name = cfg.retrieve("trainer.outcome_col_name")
    preprocessing_pipeline = cfg.retrieve("trainer.preprocessing_pipeline")

    # Set run name
    cfg.mutate("logger.*.mlflow.experiment_name", "CVD, test-set")

    # Clear unused keys from base config
    cfg = (
        cfg.mutate("trainer.@trainers", "split_trainer_separate_preprocessing")
        .remove("trainer.outcome_col_name")  # Split for train and test
        .remove("trainer.preprocessing_pipeline")  # Split for train and test)
    )

    # Handle training set setup
    cfg = (
        cfg.add("trainer.training_outcome_col_name", outcome_col_name)
        .add("trainer.training_preprocessing_pipeline", preprocessing_pipeline)
        .mutate(
            "trainer.training_preprocessing_pipeline.*.split_filter.splits_to_keep",
            ["train", "val"],
        )
    )

    # Handle validation set setup
    cfg = (
        cfg.add(  # Handle validation dataset
            "trainer.validation_data", cfg.retrieve("trainer.training_data")
        )
        .mutate("trainer.validation_outcome_col_name", outcome_col_name)
        .add("trainer.validation_preprocessing_pipeline", preprocessing_pipeline)
        .mutate("trainer.validation_preprocessing_pipeline.*.split_filter.splits_to_keep", ["test"])
    )

    train_baseline_model_from_cfg(cfg)


if __name__ == "__main__":
    populate_baseline_registry()
    populate_with_cvd_registry()

    test_set(PsycopConfig().from_disk(Path(__file__).parent / "cvd_baseline.cfg"))
