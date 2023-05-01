"""Module for filtering columns before split."""
import re
from collections.abc import Sequence

import pandas as pd
from psycop_model_training.config_schemas.data import DataSchema
from psycop_model_training.config_schemas.preprocessing import (
    PreSplitPreprocessingConfigSchema,
)
from psycop_model_training.data_loader.data_loader import msg
from psycop_model_training.utils.col_name_inference import (
    infer_look_distance,
    infer_outcome_col_name,
    infer_predictor_col_name,
)
from psycop_model_training.utils.decorators import print_df_dimensions_diff
from psycop_model_training.utils.utils import get_percent_lost


class PresSplitColFilter:
    """Class for filtering columns before split."""

    def __init__(
        self,
        pre_split_cfg: PreSplitPreprocessingConfigSchema,
        data_cfg: DataSchema,
    ):
        self.pre_split_cfg = pre_split_cfg
        self.data_cfg = data_cfg

    @print_df_dimensions_diff
    def _drop_cols_not_in_lookbehind_combination(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop predictor columns that are not in the specified combination of
        lookbehind windows.

        Args:
            dataset (pd.DataFrame): Dataset.

        Returns:
            pd.DataFrame: Dataset with dropped columns.
        """

        if not self.pre_split_cfg.lookbehind_combination:
            raise ValueError("No lookbehind_combination provided.")

        # Extract all unique lookbhehinds in the dataset predictors
        lookbehinds_in_dataset = {
            int(infer_look_distance(col)[0])
            for col in infer_predictor_col_name(df=dataset)
            if len(infer_look_distance(col)) > 0
        }

        # Convert list to set
        lookbehinds_in_spec = set(
            self.pre_split_cfg.lookbehind_combination,
        )

        # Check that all loobehinds in lookbehind_combination are used in the predictors
        if not lookbehinds_in_spec.issubset(
            lookbehinds_in_dataset,
        ):
            msg.warn(
                f"One or more of the provided lookbehinds in lookbehind_combination is/are not used in any predictors in the dataset: {lookbehinds_in_spec - lookbehinds_in_dataset}",
            )

            lookbehinds_to_keep = lookbehinds_in_spec.intersection(
                lookbehinds_in_dataset,
            )

            if not lookbehinds_to_keep:
                raise ValueError("No predictors left after dropping lookbehinds.")

            msg.warn(f"Training on {lookbehinds_to_keep}.")
        else:
            lookbehinds_to_keep = lookbehinds_in_spec

        # Create a list of all predictor columns who have a lookbehind window not in lookbehind_combination list
        cols_to_drop = [
            c
            for c in infer_predictor_col_name(df=dataset)
            if all(str(l_beh) not in c for l_beh in lookbehinds_to_keep)
        ]

        cols_to_drop = [c for c in cols_to_drop if "within" in c]
        # ? Add some specification of within_x_days indicating how to parse columns to find lookbehinds. Or, alternatively, use the column spec.

        dataset = dataset.drop(columns=cols_to_drop)
        return dataset

    @print_df_dimensions_diff
    def _drop_cols_if_exceeds_look_direction_threshold(
        self,
        dataset: pd.DataFrame,
        look_direction_threshold: float,
        direction: str,
    ) -> pd.DataFrame:
        """Drop columns if they look behind or ahead longer than a specified
        threshold.

            For example, if direction is "ahead", and n_days is 30, then the column
        should be dropped if it's trying to look 60 days ahead. This is useful
        to avoid some rows having more information than others.

            Args:
                dataset (pd.DataFrame): Dataset to process.
                look_direction_threshold (float): Number of days to look in the direction.
                direction (str): Direction to look. Allowed are ["ahead", "behind"].

        Returns:
            pd.DataFrame: Dataset without the dropped columns.
        """

        cols_to_drop = []

        n_cols_before_modification = dataset.shape[1]

        if direction == "behind":
            cols_to_process = infer_predictor_col_name(df=dataset)

            for col in cols_to_process:
                # Extract lookbehind days from column name use regex
                # E.g. "column_name_within_90_days" == 90
                # E.g. "column_name_within_90_days_fallback_NaN" == 90
                lookbehind_days_strs = re.findall(r"within_(\d+)_days", col)

                if len(lookbehind_days_strs) > 0:
                    lookbehind_days = int(lookbehind_days_strs[0])
                else:
                    msg.warn(f"Could not extract lookbehind days from {col}")
                    continue

                if lookbehind_days > look_direction_threshold:
                    cols_to_drop.append(col)

        n_cols_after_modification = dataset.shape[1]
        percent_dropped = get_percent_lost(
            n_before=n_cols_before_modification,
            n_after=n_cols_after_modification,
        )

        if n_cols_before_modification - n_cols_after_modification != 0:
            msg.info(
                f"Dropped {n_cols_before_modification - n_cols_after_modification} ({percent_dropped}%) columns because they were looking {direction} further out than {look_direction_threshold} days.",
            )

        return dataset[[c for c in dataset.columns if c not in cols_to_drop]]

    @print_df_dimensions_diff
    def _keep_unique_outcome_col_with_lookahead_days_matching_conf(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Keep only one outcome column with the same lookahead days as set in
        the config."""
        outcome_cols = infer_outcome_col_name(
            df=dataset,
            prefix=self.data_cfg.outc_prefix,
            allow_multiple=True,
        )

        col_to_drop = [
            c
            for c in outcome_cols
            if f"_{str(self.pre_split_cfg.min_lookahead_days)}_" not in c
        ]

        # If no columns to drop, return the dataset
        if not col_to_drop:
            return dataset

        df = dataset.drop(col_to_drop, axis=1)

        n_col_names = len(infer_outcome_col_name(df))
        if n_col_names > 1:
            raise ValueError(
                f"Returning {n_col_names} outcome columns, will cause problems during eval.",
            )

        return df

    @staticmethod
    def _drop_datetime_columns(
        pred_prefix: str,
        dataset: pd.DataFrame,
        drop_dtypes: tuple = ("datetime64[ns]", "<M8[ns]"),
    ) -> pd.DataFrame:
        """Drop all datetime columns from the dataset."""
        columns_to_drop = [
            c for c in dataset.columns if dataset[c].dtype in drop_dtypes
        ]
        columns_to_drop = [c for c in columns_to_drop if c.startswith(pred_prefix)]

        return dataset[[c for c in dataset.columns if c not in columns_to_drop]]

    @print_df_dimensions_diff
    def n_outcome_col_names(self, df: pd.DataFrame) -> int:
        """How many outcome columns there are in a dataframe."""
        return len(infer_outcome_col_name(df=df, allow_multiple=True))

    def run_filter(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Filter a dataframe based on the config."""
        for direction in ("ahead", "behind"):
            if direction == "ahead":
                n_days = self.pre_split_cfg.min_lookahead_days
            elif direction == "behind":
                if isinstance(self.pre_split_cfg.lookbehind_combination, Sequence):
                    n_days = max(self.pre_split_cfg.lookbehind_combination)
                else:
                    n_days = self.pre_split_cfg.lookbehind_combination
            else:
                raise ValueError(f"Unknown direction {direction}")

            if n_days is not None:
                dataset = self._drop_cols_if_exceeds_look_direction_threshold(
                    dataset=dataset,
                    look_direction_threshold=n_days,
                    direction=direction,
                )

        if self.pre_split_cfg.lookbehind_combination:
            dataset = self._drop_cols_not_in_lookbehind_combination(dataset=dataset)

        if self.pre_split_cfg.keep_only_one_outcome_col:
            dataset = self._keep_unique_outcome_col_with_lookahead_days_matching_conf(
                dataset=dataset,
            )

        if self.pre_split_cfg.drop_datetime_predictor_columns:
            dataset = self._drop_datetime_columns(
                pred_prefix=self.data_cfg.pred_prefix,
                dataset=dataset,
            )

        return dataset
