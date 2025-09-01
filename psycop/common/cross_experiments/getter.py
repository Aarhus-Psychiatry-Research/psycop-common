from abc import ABC, abstractmethod

import pandas as pd
from confection import Config


class Getter(ABC):
    @staticmethod
    @abstractmethod
    def get_eval_df() -> pd.DataFrame: ...

    @staticmethod
    @abstractmethod
    def get_feature_set_df() -> pd.DataFrame: ...

    @staticmethod
    @abstractmethod
    def get_cfg() -> Config: ...
