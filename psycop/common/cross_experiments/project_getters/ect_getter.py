from pathlib import Path

import pandas as pd
from confection import Config

from psycop.common.cross_experiments.getter import Getter
from psycop.common.model_training_v2.config.config_utils import PsycopConfig

FEATURE_SETS = {
    "structured_only": [
        "basic",
        "contacts",
        "ham-broset",
        "diagnoses",
        "medication",
        "leave-suicide",
    ],
    "text_only": ["text"],
    "structured_text": [
        "basic",
        "contacts",
        "ham-broset",
        "diagnoses",
        "medication",
        "leave-suicide",
        "text",
    ],
}


class ECTGetter(Getter):
    predicted_positive_rate: float = 0.02
    n_trials=200
    n_jobs=10

    @staticmethod
    def get_eval_df() -> pd.DataFrame:
        eval_df_path = "E:/shared_resources/ect/eval_runs/ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_test/eval_df.parquet"

        return pd.read_parquet(eval_df_path)

    @staticmethod
    def get_feature_set_df() -> pd.DataFrame:
        feature_set_df_path = "E:/shared_resources/ect/feature_set/flattened_datasets/ect_feature_set/ect_feature_set.parquet"

        return pd.read_parquet(feature_set_df_path)

    @staticmethod
    def get_cfg() -> PsycopConfig:
        experiment = "ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter"
        experiment_path = (
            f"E:/shared_resources/ect/eval_runs/{experiment}_best_run_evaluated_on_test"
        )
        return PsycopConfig(Config().from_disk(path=Path(experiment_path) / "config.cfg"))

    @staticmethod
    def get_hyperparameter_tuning_cfg() -> PsycopConfig:
        config_path = "E:/frihae/psycop-common/psycop/projects/optimization_experiments/configs/ect.cfg" # TODO fh

        cfg = PsycopConfig(Config().from_disk(path=Path(config_path)))

        cfg.mut(
            "trainer.task.task_pipe.sklearn_pipe.*.model",
            {"@estimator_steps_suggesters": "xgboost_suggester"},
        )

        # Set run name
        for feature_set, features in FEATURE_SETS.items():
            cfg.mut(
                "logger.*.mlflow.experiment_name",
                f"ECT-trunc-and-hp-{feature_set}-xgboost-no-lookbehind-filter",
            )

            cfg.mut(
                "logger.*.disk_logger.run_path",
                f"E:/shared_resources/ect/training/ECT-trunc-and-hp-{feature_set}-xgboost-no-lookbehind-filter",
            )

            layer_regex = "|".join(features)

            cfg.mut(
                "trainer.preprocessing_pipeline.*.layer_selector.keep_matching",
                f".+_layer_({layer_regex}).+",
            )

            return cfg



if __name__ == "__main__":
    getter = ECTGetter()
    print(getter.get_cfg())
    print(getter.get_eval_df().head())
    print(getter.get_feature_set_df().head())
