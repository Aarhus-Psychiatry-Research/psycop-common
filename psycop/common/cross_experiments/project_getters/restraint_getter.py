from pathlib import Path

import pandas as pd
from confection import Config

from psycop.common.cross_experiments.getter import Getter
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


class RestraintGetter(Getter):
    @staticmethod
    def get_eval_df() -> pd.DataFrame:
        eval_df_path = "E:/shared_resources/restraint/eval_runs/restraint_all_tuning_v2_best_run_evaluated_on_test/eval_df.parquet"

        return pd.read_parquet(eval_df_path)

    @staticmethod
    def get_feature_set_df() -> pd.DataFrame:
        feature_set_df_path = "E:/shared_resources/restraint/flattened_datasets/full_feature_set_structured_tfidf_750_all_outcomes/full_with_pred_adm_day_count.parquet"

        return pd.read_parquet(feature_set_df_path)

    @staticmethod
    def get_cfg() -> Config:
        config_path = "E:/shared_resources/restraint/eval_runs/restraint_all_tuning_v2_best_run_evaluated_on_test/config.cfg"

        return PsycopConfig(Config().from_disk(path=Path(config_path)))


if __name__ == "__main__":
    getter = RestraintGetter()
    print(getter.get_cfg())
    print(getter.get_eval_df().head())
    print(getter.get_feature_set_df().head())
