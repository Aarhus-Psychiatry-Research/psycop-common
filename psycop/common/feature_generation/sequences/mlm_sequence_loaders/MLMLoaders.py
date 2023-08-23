import datetime
from abc import ABC, abstractmethod
from collections.abc import Sequence

import polars as pl

from psycop.common.feature_generation.loaders.raw.load_ids import load_ids
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
    Patient,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.source_event_dataframe_unpacker import (
    EventDataFramesToPatients,
    PatientColumnNames,
)


class EventDfLoader(ABC):
    @abstractmethod
    def load_events(self) -> pl.LazyFrame:
        ...


class DiagnosisLoader(EventDfLoader):
    """Load all diagnoses for all patients."""

    def load_events(self) -> pl.LazyFrame:
        """Load all a-diagnoses"""
        query = "SELECT dw_ek_borger, datotid_slut, diagnosegruppestreng FROM [fct].[FOR_kohorte_indhold_pt_journal_psyk_somatik_inkl_2021_feb2022] WHERE datotid_slut IS NOT NULL"
        df = pl.from_pandas(sql_load(query, database="USR_PS_Forsk")).lazy()
        return self.format_diagnosis_columns(df=df)

    def format_diagnosis_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = (
            df.with_columns(
                pl.col(
                    "diagnosegruppestreng",
                )  # Each row looks like A:DF432#B:DF232#Z:ALFC3 etc.
                .str.split("#")
                .alias("value"),
            )
            .drop("diagnosegruppestreng")
            .explode("value")
            .with_columns(
                [
                    pl.col("value")
                    .str.slice(offset=2)  # Remove prefix, e.g. A: or B: or Z:
                    .str.strip()  # Replace trailing whitespaces
                    .str.replace(
                        "^D",
                        "",
                    ),  # In the diagnosis DF432, the D is only in the Danish system and doesn't carry meaning. Remove it.
                    pl.col("value")
                    .str.slice(offset=0, length=1)
                    .alias("type"),  # Exctract the type, e.g. for A:DF2, extract A
                    pl.lit("diagnosis").alias(
                        "source",
                    ),  # Add a source column, indicating diagnoses
                ],
            )
            .rename({"datotid_slut": "timestamp"})
        )

        return df


class MLMDataLoader:
    @staticmethod
    def get_train_set(loaders: Sequence[EventDfLoader]) -> list[Patient]:
        event_data = pl.concat([loader.load_events() for loader in loaders])
        train_ids = pl.from_pandas(load_ids(split="train")).lazy()

        events_from_train = train_ids.join(event_data, on="dw_ek_borger", how="left")
        events_after_2013 = events_from_train.filter(
            pl.col("timestamp") > datetime.datetime(2013, 1, 1),
        )

        unpacked_patients = EventDataFramesToPatients(
            column_names=PatientColumnNames(),
        ).unpack(source_event_dataframes=[events_after_2013.collect()])

        return unpacked_patients


if __name__ == "__main__":
    patients = MLMDataLoader.get_train_set(loaders=[DiagnosisLoader()])
