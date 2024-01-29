import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

import polars as pl

from psycop.common.data_structures.patient import Patient
from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.feature_generation.sequences.event_loader import EventLoader
from psycop.common.feature_generation.sequences.patient_slice_from_events import (
    PatientSliceFromEvents,
)

from ...model_training_v2.trainer.preprocessing.step import PresplitStep
from ...sequence_models.registry import SequenceRegistry


def keep_if_min_n_visits(df: pl.LazyFrame, n_visits: Union[int, None]) -> pl.LazyFrame:
    df = df.filter(pl.col("timestamp").unique().count().over("dw_ek_borger") >= n_visits)

    return df


@SequenceRegistry.datasets.register("patient_loader")
@dataclass(frozen=True)
class PatientLoader:
    event_loaders: Sequence[EventLoader]
    split_filter: PresplitStep
    min_n_events: int | None = None
    fraction: float = 1.0

    @staticmethod
    def load_date_of_birth_df() -> pl.DataFrame:
        df = pl.from_pandas(birthdays()).rename({"date_of_birth": "timestamp"})

        return df

    def get_patients(self) -> Sequence[Patient]:
        event_data = pl.concat([loader.load_events() for loader in self.event_loaders])

        if self.min_n_events:
            event_data = keep_if_min_n_visits(event_data, n_visits=self.min_n_events)

        rows_in_split = self.split_filter.apply(event_data)

        events_after_2013 = rows_in_split.filter(
            pl.col("timestamp") > datetime.datetime(2013, 1, 1)
        )

        unpacked_patients = PatientSliceFromEvents().unpack(
            source_event_dataframes=[events_after_2013.collect()],
            date_of_birth_df=PatientLoader.load_date_of_birth_df(),
        )

        return unpacked_patients
