"""Class for filtering prediction times before they are used for feature
generation."""
from typing import Optional

import pandas as pd


class PredictionTimeFilterer:
    """Class for filtering prediction times before they are used for
    feature."""

    def __init__(
        self,
        prediction_time_df: pd.DataFrame,
        id_col_name: str,
        quarantine_df: Optional[pd.DataFrame] = None,
        quarantine_days: Optional[int] = None,
    ):
        self.prediction_time_df = prediction_time_df
        self.quarantine_df = quarantine_df
        self.quarantine_days = quarantine_days
        self.id_col_name = id_col_name

    def _filter_prediction_times_by_quarantine_period(self):
        # We need to check if ANY quarantine date hits each prediction time.
        # Create combinations
        df = self.prediction_time_df.merge(
            self.quarantine_df,
            on=self.id_col_name,
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
        df = self.prediction_time_df

        if self.quarantine_df is not None or self.quarantine_days is not None:
            if self.quarantine_days is None or self.quarantine_days is None:
                raise ValueError(
                    "If either of quarantine_df and quarantine_days are provided, both must be provided.",
                )

            df = self._filter_prediction_times_by_quarantine_period()

        return df
