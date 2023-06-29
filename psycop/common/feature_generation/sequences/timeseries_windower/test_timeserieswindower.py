import datetime
import random
from collections.abc import Sequence
from typing import TypeVar

import polars as pl
import pytest

from psycop.common.feature_generation.sequences.timeseries_windower.types.event_dataframe import (
    EventDataframeBundle,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.prediction_time_dataframe import (
    PredictionTimeColumns,
    PredictiontimeDataframeBundle,
    create_pred_time_uuids,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.sequence_dataframe import (
    SequenceDataframeBundle,
    SequenceDataframeColumns,
)
from psycop.common.test_utils.str_to_df import str_to_pl_df

PolarsFrame = TypeVar("PolarsFrame", pl.DataFrame, pl.LazyFrame)

from wasabi import Printer

msg = Printer(timestamp=True)


def create_random_timestamps_series(
    n_rows: int,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
    name: str = "timestamp",
) -> pl.Series:
    start_timestamp_int = int(start_datetime.timestamp())
    end_timestamp_int = int(end_datetime.timestamp())

    return pl.Series(
        name=name,
        values=[
            datetime.datetime.fromtimestamp(
                random.uniform(start_timestamp_int, end_timestamp_int),
            )
            for _ in range(n_rows)
        ],
    )


@pytest.fixture(scope="module")
def c() -> SequenceDataframeColumns:
    return SequenceDataframeColumns()


class TestTimeserieswindower:
    @staticmethod
    def test_windowing_should_be_fast(
        c: SequenceDataframeColumns,
        n_patients: int = 1000,
        n_event_dfs: int = 15,
        events_per_patient: int = 50,
    ):
        ##############
        # Test setup #
        ##############
        # Create transaction date in range
        d1 = datetime.datetime.strptime("1/1/2100", "%m/%d/%Y")
        d2 = datetime.datetime.strptime("1/1/2105", "%m/%d/%Y")

        msg.info("Generating prediction times...")
        prediction_times_df_with_timestamps = pl.DataFrame(
            {
                c.entity_id: list(range(n_patients)),
            },
        )

        prediction_times_df_with_timestamps = (
            prediction_times_df_with_timestamps.with_columns(
                create_random_timestamps_series(
                    n_rows=n_patients,
                    start_datetime=d1,
                    end_datetime=d2,
                    name=c.pred_timestamp,
                ),
            )
        )

        prediction_times_bundle = PredictiontimeDataframeBundle(
            _df=prediction_times_df_with_timestamps.lazy(),
            _cols=PredictionTimeColumns(),
        )

        event_bundles = []

        n_events_per_df = n_patients * events_per_patient

        msg.info(
            f"Generating {n_event_dfs} event dataframes with {n_events_per_df} events per df..."
        )

        df = pl.DataFrame(
            {c.entity_id: list(range(n_patients)) * events_per_patient}
        ).with_columns(
            create_random_timestamps_series(
                n_rows=n_events_per_df,
                start_datetime=d1,
                end_datetime=d2,
                name=c.event_timestamp,
            ),
            pl.lit("type").alias(c.event_type),
            pl.lit("source").alias(c.event_source),
            pl.Series([random.random() for _ in range(n_events_per_df)]).alias(
                c.event_value
            ),
        )

        for i in range(n_event_dfs):
            msg.info(f"Generating event dataframe {i}...")

            event_bundles.append(EventDataframeBundle(_df=df.lazy()))

        ##################
        # Actual testing #
        ##################

        windowed = window_timeseries(
            prediction_times_bundle=prediction_times_bundle, event_bundles=event_bundles
        )

        df, cols = windowed.unpack()

        start_time = datetime.datetime.now()
        msg.info("Collecting windowed dataframe...")
        df_collected = df.collect()
        end_time = datetime.datetime.now()

        duration_seconds = (end_time - start_time).total_seconds()
        msg.info(f"Windowing took: {duration_seconds} seconds")

        assert len(df_collected) == n_patients * n_event_dfs * events_per_patient

    @staticmethod
    def test_event_dataframes_should_add_events_x_prediction_times_rows(
        c: SequenceDataframeColumns,
    ):
        prediction_times_df = str_to_pl_df(
            f"""{c.entity_id},{c.pred_timestamp},
            1,2021-01-10 00:00:00,
            2,2021-01-10 00:00:00,""",
        ).with_columns(
            create_pred_time_uuids(
                entity_id_col_name=f"{c.entity_id}",
                timestamp_col_name=f"{c.pred_timestamp}",
            ).alias("pred_time_uuid"),
        )

        bp_events = str_to_pl_df(
            f"""{c.entity_id},{c.event_type},{c.event_timestamp},{c.event_source},{c.event_value},
            1,bp,2021-01-10 00:00:00,lab,1,
            1,bp,2021-01-10 00:00:01,lab,0,
            """,
        )

        hba1c_events = str_to_pl_df(
            f"""{c.entity_id},{c.event_type},{c.event_timestamp},{c.event_source},{c.event_value},
            1,hba1c,2021-01-10 00:00:00,lab,1,
            1,hba1c,2021-01-10 00:00:01,lab,0,
        """,
        )

        result_df, result_cols = window_timeseries(
            prediction_times_bundle=PredictiontimeDataframeBundle(
                _df=prediction_times_df.lazy(),
            ),
            event_bundles=[
                EventDataframeBundle(_df=bp_events.lazy()),
                EventDataframeBundle(_df=hba1c_events.lazy()),
            ],
        ).unpack()

        assert len(result_df.collect()) == 4

    @staticmethod
    def test_windowing_should_cut_between_timestamps(c: SequenceDataframeColumns):
        prediction_times_df_bundle = PredictiontimeDataframeBundle(
            _df=str_to_pl_df(
                f"""{c.entity_id},{c.pred_timestamp},
            1,2020-01-10 00:00:00,
            """,
            ).lazy(),
        )
        lookbehind = datetime.timedelta(days=1)

        events = EventDataframeBundle(
            _df=str_to_pl_df(
                f"""{c.entity_id},{c.event_type},{c.event_source},{c.event_timestamp},{c.event_value},
            1,bp,bedside,2020-01-09 00:00:00,1, # Dropped
            1,bp,bedside,2020-01-09 00:00:01,1, # Kept
            """,
            ).lazy(),
        )

        result_df, _ = window_timeseries(
            prediction_times_bundle=prediction_times_df_bundle,
            event_bundles=[events],
            lookbehind=lookbehind,
        ).unpack()

        assert len(result_df.collect()) == 1


def window_timeseries(
    prediction_times_bundle: PredictiontimeDataframeBundle,
    event_bundles: Sequence[EventDataframeBundle],
    lookbehind: datetime.timedelta | None = None,
) -> SequenceDataframeBundle:
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
        _df=pl.concat(exploded_dfs).drop_nulls(),
        _cols=SequenceDataframeColumns(),
    )
