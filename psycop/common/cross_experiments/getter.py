from abc import ABC, abstractmethod

from confection import Config
import pandas as pd


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
