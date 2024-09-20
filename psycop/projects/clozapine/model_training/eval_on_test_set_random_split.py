from joblib import Parallel, delayed

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry


def eval_random_split_test_set(cfg: PsycopConfig, feature_set: str):
    outcome_col_name = cfg.retrieve("trainer.outcome_col_name")
    preprocessing_pipeline = cfg.retrieve("trainer.preprocessing_pipeline")

    # Setup for test set
    cfg = (
        cfg.mut("logger.*.mlflow.experiment_name", "clozapine random split test set, xgboost")
        .add("logger.*.mlflow.run_name", f"{feature_set}")
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

    feature_sets = ["structured_only", "text_only", "structured_text"]
    cfgs = [
        MlflowClientWrapper()
        .get_best_run_from_experiment(
            experiment_name=f"clozapine hparam, {feature_set}, xgboost, no lookbehind filter",
            larger_is_better=True,
            metric="all_oof_BinaryAUROC",
        )
        .get_config()
        for feature_set in feature_sets
    ]
    Parallel(n_jobs=len(feature_sets))(
        delayed(eval_random_split_test_set)(cfg=cfg, feature_set=feature_set)
        for cfg, feature_set in zip(cfgs, feature_sets)
    )
