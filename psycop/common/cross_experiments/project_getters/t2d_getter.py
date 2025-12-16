from pathlib import Path

import pandas as pd
from confection import Config

from psycop.common.cross_experiments.getter import Getter
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


class T2DGetter(Getter):
    predicted_positive_rate: float = 0.03

    @staticmethod
    def get_eval_df() -> pd.DataFrame:
        experiment = "nonviolentstigmaria-eval-on-test"
        eval_df_path = f"E:/shared_resources/t2d/model_eval/urosepsis-helicoid-eval-on-test/{experiment}_best_run_evaluated_on_test/evaluation_dataset.parquet"

        return pd.read_parquet(eval_df_path)

    @staticmethod
    def get_feature_set_df() -> pd.DataFrame:
        feature_set_df_path = "psycop_t2d_adminber_features_2023_04_27_14_25"

        all_files = Path(feature_set_df_path).glob("*.parquet")
        df_list = [pd.read_parquet(file) for file in all_files]
        feature_set_df = pd.concat(df_list, ignore_index=True)

        return feature_set_df

    @staticmethod
    def get_cfg() -> PsycopConfig:
        experiment = "nonviolentstigmaria-eval-on-test"
        experiment_path = f"E:/shared_resources/t2d/model_eval/urosepsis-helicoid-eval-on-test/{experiment}_best_run_evaluated_on_test"
        return PsycopConfig(Config().from_disk(path=Path(experiment_path) / "config.cfg"))


if __name__ == "__main__":
    getter = T2DGetter()
    print(getter.get_cfg())
    print(getter.get_eval_df().head())
    print(getter.get_feature_set_df().head())
