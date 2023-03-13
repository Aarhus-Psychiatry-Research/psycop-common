"""Pre-split value transformer. These transformations are applied before the
split.

To avoid test/train leakage, the transformations must not use any
information about the values in the dataset.
"""
from datetime import datetime

import pandas as pd
from psycop_model_training.config_schemas.data import DataSchema
from psycop_model_training.config_schemas.preprocessing import (
    PreSplitPreprocessingConfigSchema,
)
from psycop_model_training.utils.col_name_inference import infer_predictor_col_name
from psycop_model_training.utils.decorators import print_df_dimensions_diff
from wasabi import Printer

msg = Printer(timestamp=True)


class PreSplitValueTransformer:
    """Pre-split value transformer."""

    def __init__(
        self,
        pre_split_cfg: PreSplitPreprocessingConfigSchema,
        data_cfg: DataSchema,
    ):
        self.pre_split_cfg = pre_split_cfg
        self.data_cfg = data_cfg

    @print_df_dimensions_diff
    def _convert_boolean_dtypes_to_int(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert boolean dtypes to int."""
        for col in dataset.columns:
            if dataset[col].dtype == bool:
                dataset[col] = dataset[col].astype(int)

        return dataset

    def _convert_datetimes_to_ordinal(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns to integers."""
        datetime_dtypes = {"datetime64[ns]", "<M8[ns]"}

        dt_columns = [
            c for c in dataset.columns if dataset[c].dtypes in datetime_dtypes
        ]

        # convert datetime columns
        for colname in dt_columns:
            dataset[colname] = dataset[colname].map(datetime.toordinal)

        return dataset

    def _convert_predictors_to_boolean(
        self,
        dataset: pd.DataFrame,
        columns_to_skip: tuple[str, str] = ("age_in_years", "sex_female"),
        ignore_dtypes: tuple = ("datetime64[ns]", "<M8[ns]"),
    ) -> pd.DataFrame:
        """Convert predictors to boolean."""
        columns = infer_predictor_col_name(df=dataset, prefix=self.data_cfg.pred_prefix)

        cols_to_round = [
            c
            for c in columns
            if (dataset[c].dtype not in ignore_dtypes) or c in columns_to_skip
        ]

        for col in cols_to_round:
            dataset[col] = dataset[col].notnull()

        return dataset

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataset."""
        if self.pre_split_cfg.convert_booleans_to_int:
            dataset = self._convert_boolean_dtypes_to_int(dataset=dataset)

        if self.pre_split_cfg.convert_datetimes_to_ordinal:
            dataset = self._convert_datetimes_to_ordinal(dataset=dataset)

        if self.pre_split_cfg.convert_to_boolean:
            dataset = self._convert_predictors_to_boolean(dataset=dataset)

        msg.info("Finished processing dataset")

        return dataset
