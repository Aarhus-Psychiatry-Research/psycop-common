from pathlib import Path

import pandas as pd
from confection import Config

from psycop.common.cross_experiments.getter import Getter
from psycop.common.model_training_v2.config.config_utils import PsycopConfig


class ForcedAdmissionsInpatientGetter(Getter):
    predicted_positive_rate: float = 0.05
    n_trials: int = 300
    n_jobs: int = 10

    @staticmethod
    def get_eval_df() -> pd.DataFrame:
        """
        OBS: the path points to the eval_df generated from the config file written to train the original model but in model_training_v2 notation. It is not completely identical to the original trained model (the one reported in the paper), but it is very close. The original model eval_df is named evaluation_dataset.parquet and is located in the same folder.
        """
        eval_df_path = "E:/shared_resources/forced_admissions_inpatient/models/full_model_with_text_features_train_val/pipeline_eval/chuddahs-caterwauls-eval-on-test/abrasiometerintergradient-eval-on-test/eval_df.parquet"

        return pd.read_parquet(eval_df_path)

    @staticmethod
    def get_feature_set_df() -> pd.DataFrame:
        feature_set_df_path = "E:/shared_resources/forced_admissions_inpatient/flattened_datasets/full_feature_set_with_sentence_transformers_and_tfidf_750/full_feature_set.parquet"

        return pd.read_parquet(feature_set_df_path)

    @staticmethod
    def get_cfg() -> PsycopConfig:
        config_path = "E:/shared_resources/forced_admissions_inpatient/models/full_model_with_text_features_train_val/pipeline_eval/chuddahs-caterwauls-eval-on-test/abrasiometerintergradient-eval-on-test//config.cfg"

        # read and return config
        return PsycopConfig(Config().from_disk(path=Path(config_path)))

    @staticmethod
    def get_hyperparameter_tuning_cfg() -> PsycopConfig:
        config_path = "E:/frihae/psycop-common/psycop/projects/optimization_experiments/configs/fai.cfg" # TODO fh

        return PsycopConfig(Config().from_disk(path=Path(config_path)))


if __name__ == "__main__":
    getter = ForcedAdmissionsInpatientGetter()
    print(getter.get_eval_df().head())
    print(getter.get_feature_set_df().head())
    print(getter.get_cfg())
