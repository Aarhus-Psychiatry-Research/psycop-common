import datetime
from collections.abc import Sequence
from typing import Union

import polars as pl

from psycop.common.data_structures.patient import Patient
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.feature_generation.loaders.raw.load_ids import SplitName, load_ids
from psycop.common.feature_generation.sequences.event_loader import (
    DiagnosisLoader,
    EventLoader,
)
from psycop.common.feature_generation.sequences.patient_slice_from_events import (
    PatientSliceFromEvents,
)


def keep_if_min_n_visits(
    df: pl.LazyFrame,
    n_visits: Union[int, None],
) -> pl.LazyFrame:
    df = df.filter(
        pl.col("timestamp").unique().count().over("dw_ek_borger") >= n_visits,
    )

    return df


class PatientLoader:
    @staticmethod
    def load_date_of_birth_df() -> pl.DataFrame:
        df = pl.from_pandas(birthdays()).rename({"date_of_birth": "timestamp"})

        return df

    @staticmethod
    def get_split(
        event_loaders: Sequence[EventLoader],
        split: SplitName,
        min_n_events: int | None = None,
        fraction: float = 1.0,
    ) -> Sequence[Patient]:
        event_data = pl.concat([loader.load_events() for loader in event_loaders])
        split_ids = (
            pl.from_pandas(load_ids(split=split)).sample(fraction=fraction).lazy()
        )

        events_from_train = split_ids.join(event_data, on="dw_ek_borger", how="left")

        if min_n_events:
            events_from_train = keep_if_min_n_visits(
                events_from_train,
                n_visits=min_n_events,
            )

        events_after_2013 = events_from_train.filter(
            pl.col("timestamp") > datetime.datetime(2013, 1, 1),
        )

        unpacked_patients = PatientSliceFromEvents().unpack(
            source_event_dataframes=[events_after_2013.collect()],
            date_of_birth_df=PatientLoader.load_date_of_birth_df(),
        )

        return unpacked_patients


if __name__ == "__main__":
    patients = PatientLoader.get_split(
        event_loaders=[DiagnosisLoader()],
        split=SplitName.TRAIN,
        min_n_events=5,
    )
