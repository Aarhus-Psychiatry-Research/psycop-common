from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry
from psycop.projects.t2d_bigdata.model_training.populate_t2d_bigdata_registry import (
    populate_with_t2d_bigdata_registry,
)


def eval_geographic_test_set(cfg: PsycopConfig):
    outcome_col_name = cfg.retrieve("trainer.outcome_col_name")
    preprocessing_pipeline = cfg.retrieve("trainer.preprocessing_pipeline")

    # Setup for test set
    cfg = (
        cfg.mut("logger.*.mlflow.experiment_name", "T2D-bigdata, test-set")
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
    populate_with_t2d_bigdata_registry()

    eval_geographic_test_set(
        MlflowClientWrapper().get_run("T2D-bigdata", "kindly-squirrel-385").get_config()
    )
