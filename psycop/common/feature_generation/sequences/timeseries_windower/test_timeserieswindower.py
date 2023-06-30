import datetime
import random
from typing import List, TypeVar

import polars as pl
import pytest

from psycop.common.feature_generation.sequences.timeseries_windower.timeseries_windower import (
    window_timeseries,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.event_dataframe import (
    EventDataframeBundle,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.prediction_time_dataframe import (
    PredictiontimeColumns,
    PredictiontimeDataframeBundle,
    create_pred_time_uuids,
)
from psycop.common.feature_generation.sequences.timeseries_windower.types.sequence_dataframe import (
    SequenceColumns,
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
def c() -> SequenceColumns:
    return SequenceColumns()


def generate_prediction_times_bundles(
    c: SequenceColumns,
    n_patients: int,
    d1: datetime.datetime,
    d2: datetime.datetime,
) -> PredictiontimeDataframeBundle:
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
        df=prediction_times_df_with_timestamps.lazy(),
        cols=PredictiontimeColumns(),
    )

    return prediction_times_bundle


def generate_event_bundles(
    c: SequenceColumns,
    n_patients: int,
    n_event_dfs: int,
    events_per_patient: int,
    d1: datetime.datetime,
    d2: datetime.datetime,
    n_events_per_df: int,
) -> List[EventDataframeBundle]:
    msg.info(
        f"Generating {n_event_dfs} event dataframes with {n_events_per_df} events per df...",
    )

    df = pl.DataFrame(
        {c.entity_id: list(range(n_patients)) * events_per_patient},
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
            c.event_value,
        ),
    )

    event_bundles: List[EventDataframeBundle] = []

    for i in range(n_event_dfs):
        msg.info(f"Generating event dataframe {i}...")

        event_bundles.append(EventDataframeBundle(df=df.lazy()))

    return event_bundles


class TestTimeserieswindower:
    @staticmethod
    def test_windowing_should_be_fast(
        c: SequenceColumns,
        n_patients: int = 1000,
        n_event_dfs: int = 15,
        events_per_patient: int = 50,
    ):
        """
        Allow manual benchmarking of windowing function.
        Have decided against performance assertions since the variation in github actions runners' performance is too high.
        """
        ##############
        # Test setup #
        ##############
        # Create prediction time date-range
        d1 = datetime.datetime.strptime("1/1/2100", "%m/%d/%Y")
        d2 = datetime.datetime.strptime("1/1/2101", "%m/%d/%Y")

        prediction_times_bundle = generate_prediction_times_bundles(
            c=c, n_patients=n_patients, d1=d1, d2=d2
        )

        n_events_per_df = n_patients * events_per_patient
        event_bundles = generate_event_bundles(
            c=c,
            n_patients=n_patients,
            n_event_dfs=n_event_dfs,
            events_per_patient=events_per_patient,
            d1=d1,
            d2=d2,
            n_events_per_df=n_events_per_df,
        )

        ##################
        # Actual testing #
        ##################
        windowed = window_timeseries(
            prediction_times_bundle=prediction_times_bundle,
            event_bundles=event_bundles,
            lookbehind=datetime.timedelta(days=368),
        )

        df, cols = windowed.unpack()

        start_time = datetime.datetime.now()
        msg.info("Collecting windowed dataframe...")
        df_collected = df.collect()
        end_time = datetime.datetime.now()

        duration_seconds = (end_time - start_time).total_seconds()
        msg.info(f"Windowing took: {duration_seconds} seconds")
        # 13 seconds for 120_000 patients · 15 event dataframes · 50 events per patient
        # on an M1 MacBook Pro

        assert len(df_collected) == n_patients * n_event_dfs * events_per_patient

    @staticmethod
    def test_event_dataframes_add_correct_number_of_rows(
        c: SequenceColumns,
    ):
        prediction_times_df = str_to_pl_df(
            f"""{c.entity_id},{c.pred_timestamp},
            1,2021-01-10 00:00:00, # Included, since some events match
            2,2021-01-10 00:00:00, # Not included, since no events match
            """,
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
                df=prediction_times_df.lazy(),
            ),
            event_bundles=[
                EventDataframeBundle(df=bp_events.lazy()),
                EventDataframeBundle(df=hba1c_events.lazy()),
            ],
            lookbehind=None,
        ).unpack()

        # Ensure that the resulting dataframe is number of prediction times for an ID · number of events for that ID.
        # Prediction times with no events should be dropped. E.g. for entity_id == 2 is dropped.
        assert len(result_df.collect()) == (len(prediction_times_df) - 1) * len(
            bp_events
        ) + len(hba1c_events)

    @staticmethod
    def test_windowing_should_cut_between_timestamps(c: SequenceColumns):
        prediction_times_df_bundle = PredictiontimeDataframeBundle(
            df=str_to_pl_df(
                f"""{c.entity_id},{c.pred_timestamp},
            1,2020-01-10 00:00:00,
            """,
            ).lazy(),
        )
        lookbehind = datetime.timedelta(days=1)

        events = EventDataframeBundle(
            df=str_to_pl_df(
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
