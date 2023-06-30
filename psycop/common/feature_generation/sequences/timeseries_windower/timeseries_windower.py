import datetime
from collections.abc import Sequence

import polars as pl

from psycop.common.feature_generation.sequences.timeseries_windower.types.event_dataframe import (
    EventDataframeBundle,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.prediction_time_dataframe import (
    PredictiontimeDataframeBundle,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.sequence_dataframe import (
    SequenceColumns,
    SequenceDataframeBundle,
)


def window_timeseries(
    prediction_times_bundle: PredictiontimeDataframeBundle,
    event_bundles: Sequence[EventDataframeBundle],
    lookbehind: datetime.timedelta | None,
) -> SequenceDataframeBundle:
    """Take a list of prediction times and a list of events. Map those onto one another based on entity_id, and generate sequences with events within the lookbehind window. See also the tests for example behavior.

    Args:
        prediction_times_bundle: Prediction times. See the object for requirec columns.
        event_bundles: Events to map onto prediction times. See the object for required columns.
        lookbehind: How far back to look for events. If None, includes all events before the prediction time.

    Returns:
        SequenceDataframeBundle: The output dataframe sequence. See the object for required columns.
    """
    pred_time_df, pred_time_cols = prediction_times_bundle.unpack()
    exploded_dfs = []

    for event_bundle in event_bundles:
        event_df, event_cols = event_bundle.unpack()

        exploded_df = pred_time_df.join(event_df, on=event_cols.entity_id, how="left")

        if lookbehind is not None:
            exploded_df = exploded_df.filter(
                pl.col(event_cols.timestamp)
                > (pl.col(pred_time_cols.timestamp) - lookbehind),
            )

        exploded_dfs.append(exploded_df)

    return SequenceDataframeBundle(
        df=pl.concat(exploded_dfs).drop_nulls(),
        cols=SequenceColumns(),
    )
