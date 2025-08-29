from pathlib import Path
from confection import Config
import pandas as pd
from psycop.common.cross_experiments.getter import Getter
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


class ECTGetter(Getter):
    @staticmethod
    def get_eval_df() -> pd.DataFrame:
        eval_df_path = "E:/shared_resources/ect/ECT-trunc-and-hp-structured_text-xgboost-no-lookbehind-filter_best_run_evaluated_on_test/eval_df.parquet"

        return pd.read_parquet(eval_df_path)

    @staticmethod
    def get_feature_set_df() -> pd.DataFrame:
        feature_set_df_path = "E:/shared_resources/ect/flattened_datasets/full_feature_set_structured_tfidf_750_all_outcomes/full_with_pred_adm_day_count.parquet"

        return pd.read_parquet(feature_set_df_path)

    @staticmethod
    def get_cfg() -> Config:
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
