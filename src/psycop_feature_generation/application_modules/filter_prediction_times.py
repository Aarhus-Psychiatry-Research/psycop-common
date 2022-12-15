"""Class for filtering prediction times before they are used for feature
generation."""
from typing import Optional

import pandas as pd


class PredictionTimeFilterer:
    """Class for filtering prediction times before they are used for
    feature."""

    def __init__(
        self,
        prediction_times_df: pd.DataFrame,
        entity_id_col_name: str,
        quarantine_timestamps_df: Optional[pd.DataFrame] = None,
        quarantine_interval_days: Optional[int] = None,
    ):
        """Initialize PredictionTimeFilterer.

        Args:
            prediction_times_df (pd.DataFrame): Prediction times dataframe.
                Should contain entity_id and timestamp columns with col_names matching those in project_info.col_names.
            quarantine_df (pd.DataFrame, optional): A dataframe with "timestamp" column from which to start the quarantine.
                Any prediction times within the quarantine_interval_days after this timestamp will be dropped.
            quarantine_days (int, optional): Number of days to quarantine.
            entity_id_col_name (str): Name of the entity_id_col_name column.
        """

        self.prediction_times_df = prediction_times_df
        self.quarantine_df = quarantine_timestamps_df
        self.quarantine_days = quarantine_interval_days
        self.entity_id_col_name = entity_id_col_name

    def _filter_prediction_times_by_quarantine_period(self):
        # We need to check if ANY quarantine date hits each prediction time.
        # Create combinations
        df = self.prediction_times_df.merge(
            self.quarantine_df,
            on=self.entity_id_col_name,
            how="left",
            suffixes=("_pred", "_quarantine"),
        )

        df["days_since_quarantine"] = (
            df["timestamp_pred"] - df["timestamp_quarantine"]
        ).dt.days

        # Check if the prediction time is hit by the quarantine date.
        df.loc[
            (df["days_since_quarantine"] < self.quarantine_days)
            & (df["days_since_quarantine"] > 0),
            "hit_by_quarantine",
        ] = True

        # If any of the combinations for a UUID is hit by a quarantine date, drop it.
        df["hit_by_quarantine"] = df.groupby("pred_time_uuid")[
            "hit_by_quarantine"
        ].transform("max")

        df = df.loc[
            df["hit_by_quarantine"] != True  # pylint: disable=singleton-comparison
        ]

        df = df.drop_duplicates(subset="pred_time_uuid")

        # Drop the columns we added
        df = df.drop(
            columns=[
                "days_since_quarantine",
                "hit_by_quarantine",
                "timestamp_quarantine",
            ],
        )

        # Rename the timestamp column
        df = df.rename(columns={"timestamp_pred": "timestamp"})

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

        return df
