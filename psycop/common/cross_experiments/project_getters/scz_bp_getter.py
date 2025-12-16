from pathlib import Path

import pandas as pd
from confection import Config

from psycop.common.cross_experiments.getter import Getter
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


class ECTGetter(Getter):
    predicted_positive_rate: float = 0.02

    @staticmethod
    def get_eval_df() -> pd.DataFrame:
        eval_df_path = "E:/shared_resources/scz_bp/testing/eval_df.parquet"

        return pd.read_parquet(eval_df_path)

    @staticmethod
    def get_feature_set_df() -> pd.DataFrame:
        feature_set_df_path = "E:/shared_resources/scz_bp/flattened_datasets/l1_l4-lookbehind_183_365_730-all_relevant_tfidf_1000_lookbehind_730.parquet"

        return pd.read_parquet(feature_set_df_path)

    @staticmethod
    def get_cfg() -> PsycopConfig:
        experiment_path = "E:/shared_resources/scz_bp/testing"
        return PsycopConfig(Config().from_disk(path=Path(experiment_path) / "config.cfg"))


if __name__ == "__main__":
    getter = ECTGetter()
    print(getter.get_cfg())
    print(getter.get_eval_df().head())
    print(getter.get_feature_set_df().head())
