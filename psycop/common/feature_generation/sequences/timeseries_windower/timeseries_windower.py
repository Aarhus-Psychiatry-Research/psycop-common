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
    SequenceColumnNames,
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
    output_col_names = SequenceColumnNames()

    # Standardise column names
    pred_time_df = pred_time_df.rename(
        {
            pred_time_cols.entity_id: output_col_names.entity_id,
            pred_time_cols.pred_time_uuid: output_col_names.pred_time_uuid,
            pred_time_cols.timestamp: output_col_names.pred_timestamp,
        }
    )

    for event_bundle in event_bundles:
        event_df, event_cols = event_bundle.unpack()

        event_df = event_df.rename(
            {
                event_cols.entity_id: output_col_names.entity_id,
                event_cols.timestamp: output_col_names.event_timestamp,
                event_cols.event_type: output_col_names.event_type,
                event_cols.event_value: output_col_names.event_value,
                event_cols.event_source: output_col_names.event_source,
            }
        )

        exploded_df = pred_time_df.join(
            event_df, on=output_col_names.entity_id, how="left", suffix="_event"
        )

        if lookbehind is not None:
            exploded_df = exploded_df.filter(
                (pl.col(output_col_names.event_timestamp)
                > (pl.col(output_col_names.pred_timestamp) - lookbehind)) & (pl.col(output_col_names.event_timestamp) < pl.col(output_col_names.pred_timestamp)),
            )

        exploded_dfs.append(exploded_df)

    concatenated_dfs = pl.concat(exploded_dfs)

    return SequenceDataframeBundle(
        df=concatenated_dfs,
        cols=SequenceColumnNames(),
    )
