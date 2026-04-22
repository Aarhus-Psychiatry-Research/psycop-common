from pathlib import Path

import pandas as pd
from confection import Config

from psycop.common.cross_experiments.getter import Getter
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


class CVDGetter(Getter):
    predicted_positive_rate: float = 0.05

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


if __name__ == "__main__":
    getter = CVDGetter()
    print(getter.get_cfg())
    print(getter.get_eval_df().head())
    print(getter.get_feature_set_df().head())
