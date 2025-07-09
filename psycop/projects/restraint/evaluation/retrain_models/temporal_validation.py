from joblib import Parallel, delayed

from psycop.common.global_utils.mlflow.mlflow_data_extraction import MlflowClientWrapper
from psycop.common.model_training_v2.config.baseline_pipeline import train_baseline_model_from_cfg
from psycop.common.model_training_v2.config.config_utils import PsycopConfig
from psycop.common.model_training_v2.config.populate_registry import populate_baseline_registry


def eval_stratified_split(
    cfg: PsycopConfig, training_end_date: str, evaluation_interval: tuple[str, str]
) -> float:
    # Setup for experiment
    cfg = (
        cfg.rem("trainer.training_preprocessing_pipeline.*.temporal_col_filter")
        .mut(
            "logger.*.disk_logger.run_path",
            f"E:/shared_resources//restraint/eval_runs/temporal_validation/{cfg.retrieve('logger.*.mlflow.experiment_name')}_{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}",
        )
        .mut("logger.*.mlflow.experiment_name", "restraint_temporal_validation")
        .add(
            "logger.*.mlflow.run_name",
            f"{training_end_date}_{evaluation_interval[0]}_{evaluation_interval[1]}",
        )
    )

    # Handle training set setup
    cfg = (
        cfg.add(
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
            ["train", "val"],
        )
    )

    # Handle validation set setup
    cfg = (
        cfg.rem("trainer.validation_preprocessing_pipeline.*.temporal_col_filter")
        .mut(  # Handle validation dataset
            "trainer.validation_data", cfg.retrieve("trainer.training_data")
        )
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
            ["train", "val"],
        )
    )

    return train_baseline_model_from_cfg(cfg)


def evaluate_year(train_end_year: int) -> tuple[int, dict[int, float]]:
    evaluation_years = range(train_end_year, 22)
    year_aurocs = {
        y: eval_stratified_split(
            MlflowClientWrapper()
            .get_best_run_from_experiment(
                experiment_name="restraint_all_tuning_best_run_evaluated_on_test",
                larger_is_better=True,
                metric="BinaryAUROC",
            )
            .get_config(),
            training_end_date=f"20{train_end_year}-01-01",
            evaluation_interval=(f"20{y}-01-01", f"20{y}-12-31"),
        )
        for y in evaluation_years
    }
    return train_end_year, year_aurocs


if __name__ == "__main__":
    populate_baseline_registry()


train_end_years = range(16, 21)


# compute in parallel across train end year (15 workers)
# can be flattened to be done across evaluation years as well but, meh
results = Parallel(n_jobs=len(train_end_years))(
    delayed(evaluate_year)(train_end_year=train_end_year) for train_end_year in train_end_years
)
