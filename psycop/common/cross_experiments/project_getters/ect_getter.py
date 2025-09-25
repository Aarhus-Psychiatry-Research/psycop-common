from pathlib import Path

import pandas as pd
from confection import Config

from psycop.common.cross_experiments.getter import Getter
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


class ECTGetter(Getter):
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
        experiment = f"ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter"
        experiment_path = (
            f"E:/shared_resources/ect/eval_runs/{experiment}_best_run_evaluated_on_test"
        )
        return PsycopConfig(Config().from_disk(path=Path(experiment_path) / "config.cfg"))


if __name__ == "__main__":
    getter = ECTGetter()
    print(getter.get_cfg())
    print(getter.get_eval_df().head())
    print(getter.get_feature_set_df().head())
