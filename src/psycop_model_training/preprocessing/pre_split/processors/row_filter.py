"""Row filter for pre-split data."""
from datetime import timedelta
from typing import Union

import pandas as pd
from psycop_model_training.config_schemas.data import DataSchema
from psycop_model_training.config_schemas.preprocessing import (
    PreSplitPreprocessingConfigSchema,
)
from psycop_model_training.data_loader.data_loader import msg
from psycop_model_training.utils.decorators import print_df_dimensions_diff
from psycop_model_training.utils.utils import get_percent_lost


class PreSplitRowFilter:
    """Row filter for pre-split data."""

    def __init__(
        self,
        pre_split_cfg: PreSplitPreprocessingConfigSchema,
        data_cfg: DataSchema,
    ):
        self.pre_split_cfg = pre_split_cfg
        self.data_cfg = data_cfg

    @print_df_dimensions_diff
    def _drop_rows_if_datasets_ends_within_days(
        self,
        n_days: Union[float, timedelta],  # type: ignore
        dataset: pd.DataFrame,
        direction: str,
    ) -> pd.DataFrame:
        """Drop visits that lie within certain amount of days from end of
        dataset.

        Args:
            n_days (float): Number of days.
            dataset (pd.DataFrame): Dataset.
            direction (str): Direction to look. Allowed are ["before", "after"].

        Returns:
            pd.DataFrame: Dataset with dropped rows.
        """
        if not isinstance(n_days, timedelta):
            n_days: timedelta = timedelta(days=n_days)

        if direction not in ("ahead", "behind"):
            raise ValueError(f"Direction {direction} not supported.")

        n_rows_before_modification = dataset.shape[0]

        if direction == "ahead":
            max_datetime = dataset[self.data_cfg.col_name.pred_timestamp].max() - n_days
            before_max_dt = (
                dataset[self.data_cfg.col_name.pred_timestamp] < max_datetime
            )
            dataset = dataset[before_max_dt]
        elif direction == "behind":
            min_datetime = dataset[self.data_cfg.col_name.pred_timestamp].min() + n_days
            after_min_dt = dataset[self.data_cfg.col_name.pred_timestamp] > min_datetime
            dataset = dataset[after_min_dt]

        n_rows_after_modification = dataset.shape[0]
        percent_dropped = get_percent_lost(
            n_before=n_rows_before_modification,
            n_after=n_rows_after_modification,
        )

        if n_rows_before_modification - n_rows_after_modification != 0:
            msg.info(
                f"Dropped {n_rows_before_modification - n_rows_after_modification} ({percent_dropped}%) rows because the end of the dataset was within {n_days} of their prediction time when looking {direction} from their prediction time",
            )

        return dataset

    @print_df_dimensions_diff
    def _drop_patient_if_excluded_by_date(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop patients that have an exclusion event within the washin
        period."""
        if self.data_cfg.col_name.exclusion_timestamp is None:
            raise ValueError(
                "Exclusion timestamp column not specified in config. Cannot drop patients based on exclusion date.",
            )

        outcome_before_date = (
            dataset[self.data_cfg.col_name.exclusion_timestamp]
            < self.pre_split_cfg.drop_patient_if_exclusion_before_date
        )

        patients_to_drop = set(
            dataset[self.data_cfg.col_name.id][outcome_before_date].unique(),
        )

        dataset = dataset[~dataset[self.data_cfg.col_name.id].isin(patients_to_drop)]

        return dataset

    @print_df_dimensions_diff
    def _keep_only_if_older_than_min_age(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows that are older than the minimum age specified in the
        config."""
        return dataset[
            dataset[self.data_cfg.col_name.age] >= self.pre_split_cfg.min_age
        ]

    @print_df_dimensions_diff
    def _drop_visit_after_exclusion_timestamp(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Drop all rows where exclusion timestamp is before the prediction
        time."""
        if self.data_cfg.col_name.exclusion_timestamp is None:
            raise ValueError(
                "Exclusion timestamp column not specified in config. Cannot drop patients based on exclusion date.",
            )

        rows_to_drop = (
            dataset[self.data_cfg.col_name.pred_timestamp]
            > dataset[self.data_cfg.col_name.exclusion_timestamp]
        )

        return dataset[~rows_to_drop]

    @print_df_dimensions_diff
    def _drop_rows_after_event_time(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Drop all rows where prediction timestamp is after the outcome."""
        rows_to_drop = (
            dataset[self.data_cfg.col_name.pred_timestamp]
            > dataset[self.data_cfg.col_name.outcome_timestamp]
        )

        return dataset[~rows_to_drop]

    @print_df_dimensions_diff
    def _drop_rows_before_min_date(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[
            dataset[self.data_cfg.col_name.pred_timestamp]
            > self.pre_split_cfg.min_prediction_time_date
        ]

    def run_filter(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Run filters based on config."""
        if self.pre_split_cfg.min_prediction_time_date:
            self._drop_rows_before_min_date(dataset)

        for direction in ("ahead", "behind"):
            if direction == "ahead":
                n_days = self.pre_split_cfg.min_lookahead_days
            elif direction == "behind":
                if self.pre_split_cfg.lookbehind_combination is not None:
                    n_days = min(self.pre_split_cfg.lookbehind_combination)
                else:
                    n_days = None
            else:
                raise ValueError(f"Direction {direction} not supported.")

            if n_days is not None:
                dataset = self._drop_rows_if_datasets_ends_within_days(
                    n_days=n_days,
                    dataset=dataset,
                    direction=direction,
                )

        if self.pre_split_cfg.drop_patient_if_exclusion_before_date:
            if self.data_cfg.col_name.exclusion_timestamp is None:
                raise ValueError(
                    "Can't drop patients if exclusion timestamp is not specified in config.",
                )
            dataset = self._drop_patient_if_excluded_by_date(dataset)

        if self.pre_split_cfg.drop_visits_after_exclusion_timestamp:
            if self.data_cfg.col_name.exclusion_timestamp is None:
                raise ValueError(
                    "Can't drop visits if exclusion timestamp is not specified in config.",
                )
            dataset = self._drop_visit_after_exclusion_timestamp(dataset)

        if self.pre_split_cfg.min_age:
            dataset = self._keep_only_if_older_than_min_age(dataset)

        dataset = self._drop_rows_after_event_time(dataset=dataset)

        return dataset
