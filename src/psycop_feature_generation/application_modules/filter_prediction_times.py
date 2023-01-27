"""Class for filtering prediction times before they are used for feature
generation."""
import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


class PredictionTimeFilterer:
    """Class for filtering prediction times before they are used for
    feature."""

    def __init__(
        self,
        prediction_times_df: pd.DataFrame,
        entity_id_col_name: str,
        quarantine_timestamps_df: Optional[pd.DataFrame] = None,
        quarantine_interval_days: Optional[int] = None,
        timestamp_col_name: Optional[str] = "timestamp",
    ):
        """Initialize PredictionTimeFilterer.

        Args:
            prediction_times_df (pd.DataFrame): Prediction times dataframe.
                Should contain entity_id and timestamp columns with col_names matching those in project_info.col_names.
            quarantine_df (pd.DataFrame, optional): A dataframe with timestamp column from which to start the quarantine.
                Any prediction times within the quarantine_interval_days after this timestamp will be dropped.
            quarantine_days (int, optional): Number of days to quarantine.
            entity_id_col_name (str): Name of the entity_id_col_name column.
            timestamp_col_name (str, optional): Name of the timestamp column.
        """

        self.prediction_times_df = prediction_times_df
        self.quarantine_df = quarantine_timestamps_df
        self.quarantine_days = quarantine_interval_days
        self.quarantine_df = self.quarantine_df.rename(
            columns={"timestamp": "timestamp_quarantine"},
        )
        self.entity_id_col_name = entity_id_col_name
        self.timestamp_col_name = timestamp_col_name

        self.added_pred_time_uuid_col: bool = False
        self.pred_time_uuid_col_name = "pred_time_uuid"

        uuid_cols = [c for c in self.prediction_times_df.columns if "uuid" in c]

        if len(uuid_cols) == 0:
            self.added_pred_time_uuid_col = True

            self.prediction_times_df[
                self.pred_time_uuid_col_name
            ] = self.prediction_times_df[self.entity_id_col_name].astype(
                str,
            ) + self.prediction_times_df[
                timestamp_col_name
            ].dt.strftime(
                "-%Y-%m-%d-%H-%M-%S",
            )

    def _filter_prediction_times_by_quarantine_period(self):
        # We need to check if ANY quarantine date hits each prediction time.
        # Create combinations
        n_before = len(self.prediction_times_df)

        df = self.prediction_times_df.merge(
            self.quarantine_df,
            on=self.entity_id_col_name,
            how="left",
        )

        df["days_since_quarantine"] = (
            df[self.timestamp_col_name] - df["timestamp_quarantine"]
        ).dt.days

        # Check if the prediction time is hit by the quarantine date.
        df.loc[
            (df["days_since_quarantine"] < self.quarantine_days)
            & (df["days_since_quarantine"] > 0),
            "hit_by_quarantine",
        ] = True

        # Get only the rows that were hit by the quarantine date
        df_hit_by_quarantine = df.loc[
            df["hit_by_quarantine"] == True  # pylint: disable=singleton-comparison
        ].drop_duplicates(subset=[self.pred_time_uuid_col_name])[
            ["pred_time_uuid", "hit_by_quarantine"]
        ]

        # Use these rows to filter the prediction times
        df = self.prediction_times_df.merge(
            df_hit_by_quarantine,
            on=self.pred_time_uuid_col_name,
            how="left",
            suffixes=("", "_hit_by_quarantine"),
            validate="one_to_one",
        )

        df = df.loc[
            df["hit_by_quarantine"] != True  # pylint: disable=singleton-comparison
        ]

        # Drop the columns we added
        df = df.drop(
            columns=[
                "hit_by_quarantine",
            ],
        )

        n_after = len(df)
        log.info(
            f"Filtered {n_before - n_after} prediction times by quarantine period.",
        )

        return df

    def filter(self):
        """Run filters based on the provided parameters."""
        df = self.prediction_times_df

        if self.quarantine_df is not None or self.quarantine_days is not None:
            if all([v is None for v in (self.quarantine_days, self.quarantine_df)]):
                raise ValueError(
                    "If either of quarantine_df and quarantine_days are provided, both must be provided.",
                )

            df = self._filter_prediction_times_by_quarantine_period()

        if self.added_pred_time_uuid_col:
            df = df.drop(columns=[self.pred_time_uuid_col_name])

        return df
