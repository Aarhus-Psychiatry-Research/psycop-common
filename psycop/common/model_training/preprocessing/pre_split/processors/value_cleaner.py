"""Class for formatting values before split, e.g. assigning datetime, removing
negative values etc."""
import numpy as np
import pandas as pd

from psycop.common.model_training.config_schemas.data import DataSchema
from psycop.common.model_training.config_schemas.preprocessing import (
    PreSplitPreprocessingConfigSchema,
)
from psycop.common.model_training.utils.col_name_inference import (
    infer_predictor_col_name,
)
from psycop.common.model_training.utils.decorators import print_df_dimensions_diff


class PreSplitValueCleaner:
    """Class for cleaning values before split, e.g. assigning datetime,
    removing negative values etc."""

    def __init__(
        self,
        pre_split_cfg: PreSplitPreprocessingConfigSchema,
        data_cfg: DataSchema,
    ):
        self.pre_split_cfg = pre_split_cfg
        self.data_cfg = data_cfg

    @staticmethod
    @print_df_dimensions_diff
    def convert_timestamp_dtype_and_nat(dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert columns with `timestamp`in their name to datetime, and
        convert 0's to NaT."""
        timestamp_colnames = [col for col in dataset.columns if "timestamp" in col]

        for colname in timestamp_colnames:
            if dataset[colname].dtype != "datetime64[ns]":
                # Convert all 0s in colname to NaT
                dataset[colname] = dataset[colname].apply(
                    lambda x: pd.NaT if x == "0" else x,  # type: ignore
                )  # type: ignore
                dataset[colname] = pd.to_datetime(dataset[colname])

        return dataset

    @print_df_dimensions_diff
    def _negative_values_to_nan(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert negative values to NaN."""
        preds = dataset[infer_predictor_col_name(df=dataset)]

        # Get all columns with negative values
        cols_with_numerical_values = preds.select_dtypes(include=["number"]).columns

        numerical_columns_with_negative_values = [
            c for c in cols_with_numerical_values if preds[c].min() < 0
        ]

        df_to_replace = dataset[numerical_columns_with_negative_values].copy()

        # Convert to NaN
        df_to_replace[df_to_replace < 0] = np.nan
        dataset[numerical_columns_with_negative_values] = df_to_replace

        return dataset

    @staticmethod
    def _offset_so_no_negative_values(dataset: pd.DataFrame) -> pd.DataFrame:
        """Offset values with minimum negative value, so all values will be non-negative"""

        preds = dataset[infer_predictor_col_name(df=dataset)]

        # Get all columns with negative values
        cols_with_numerical_values = preds.select_dtypes(include=["number"]).columns

        numerical_columns_with_negative_values = [
            c for c in cols_with_numerical_values if preds[c].min() < 0
        ]

        df_to_replace = dataset[numerical_columns_with_negative_values].copy()

        # Get minimum value in each column
        col_min_values = {c: df_to_replace[c].min() for c in df_to_replace}

        # Offset values with abs min, so min becomes 0
        df_to_replace = pd.concat(
            [
                df_to_replace[c_df] + abs(col_min_values[c])
                for c_df, c in zip(df_to_replace, col_min_values)
            ],
            axis=1,
        )

        dataset[numerical_columns_with_negative_values] = df_to_replace

        return dataset

    def clean(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Apply the cleaning functions to the dataset."""
        # Super hacky transformation of negative weights (?!) for chi-square.
        # In the future, we want to:
        # 1a. See if there's a way of using feature selection that permits negative values, or
        # 1b. Always use z-score normalisation?

        if self.pre_split_cfg.negative_values_to_nan:
            dataset = self._negative_values_to_nan(dataset=dataset)

        if self.pre_split_cfg.offset_so_no_negative_values:
            dataset = self._offset_so_no_negative_values(dataset=dataset)

        dataset = self.convert_timestamp_dtype_and_nat(dataset=dataset)

        return dataset
