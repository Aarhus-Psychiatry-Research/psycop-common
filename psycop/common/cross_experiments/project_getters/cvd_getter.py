from pathlib import Path

import pandas as pd
from confection import Config

from psycop.common.cross_experiments.getter import Getter
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


class CVDGetter(Getter):
    predicted_positive_rate: float = 0.05
    n_trials: int = 150
    n_jobs: int = 10

    @staticmethod
    def get_eval_df() -> pd.DataFrame:
        experiment = "CVD-hyperparam-tuning-layer-2-xgboost-disk-logged"
        eval_df_path = f"E:/shared_resources/cvd/eval_runs/{experiment}_best_run_evaluated_on_test/eval_df.parquet"

        return pd.read_parquet(eval_df_path)

    @staticmethod
    def get_feature_set_df() -> pd.DataFrame:
        feature_set_df_path = "E:/shared_resources/cvd/feature_set/flattened_datasets/cvd_feature_set/cvd_feature_set.parquet"

        return pd.read_parquet(feature_set_df_path)

    @staticmethod
    def get_cfg() -> PsycopConfig:
        experiment = "CVD-hyperparam-tuning-layer-2-xgboost-disk-logged"
        experiment_path = (
            f"E:/shared_resources/cvd/eval_runs/{experiment}_best_run_evaluated_on_test"
        )
        return PsycopConfig(Config().from_disk(path=Path(experiment_path) / "config.cfg"))

    @staticmethod
    def get_hyperparameter_tuning_cfg() -> PsycopConfig:
        config_path = "E:/frihae/psycop-common/psycop/projects/optimization_experiments/configs/cvd.cfg" # TODO fh

        cfg = PsycopConfig(Config().from_disk(path=Path(config_path)))

        cfg.mut(
            "trainer.task.task_pipe.sklearn_pipe.*.model",
            {"@estimator_steps_suggesters": "xgboost_suggester"},
        )

        # Set run name
        for i in reversed([1, 2, 3, 4]):
            cfg.mut(
                "logger.*.mlflow.experiment_name",
                f"CVD-hyperparam-tuning-layer-{i}-xgboost-disk-logged",
            )

            cfg.mut(
                "logger.*.disk_logger.run_path",
                f"E:/shared_resources/cvd/training/CVD-hyperparam-tuning-layer-{i}-xgboost-disk-logged",
            )

            layer_regex = "|".join([str(i) for i in range(1, i + 1)])

            cfg.mut(
                "trainer.preprocessing_pipeline.*.layer_selector.keep_matching",
                f".+_layer_({layer_regex}).+",
            )

        return cfg


if __name__ == "__main__":
    getter = CVDGetter()
    print(getter.get_cfg())
    print(getter.get_eval_df().head())
    print(getter.get_feature_set_df().head())
