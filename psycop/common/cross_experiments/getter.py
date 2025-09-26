from abc import ABC, abstractmethod

import pandas as pd

from psycop.common.model_training_v2.config.config_utils import PsycopConfig


class Getter(ABC):
    @staticmethod
    @abstractmethod
    def get_eval_df() -> pd.DataFrame: ...

    @staticmethod
    @abstractmethod
    def get_feature_set_df() -> pd.DataFrame: ...

    @staticmethod
    @abstractmethod
    def get_cfg() -> PsycopConfig: ...
