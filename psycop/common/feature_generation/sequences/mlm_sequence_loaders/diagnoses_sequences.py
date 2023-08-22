from abc import ABC, abstractmethod
from collections.abc import Sequence

import polars as pl

from psycop.common.feature_generation.loaders.raw.load_ids import load_ids
from psycop.common.feature_generation.loaders.raw.sql_load import sql_load
from psycop.common.feature_generation.sequences.mlm_sequence_loaders.sequences_for_mlm import (
    AbstractMLMDataLoader,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.patient import (
    Patient,
)
from psycop.common.feature_generation.sequences.timeseries_windower_python.source_event_dataframe_unpacker import (
    PatientColumnNames,
    SourceEventDataframeUnpacker,
)


class EventDfLoader(ABC):
    @abstractmethod
    def load_events(self) -> pl.LazyFrame:
        ...


class DiagnosisLoader(EventDfLoader):
    """Load all diagnoses for all patients."""

    def load_events(self) -> pl.LazyFrame:
        """Load all a-diagnoses"""
        query = "SELECT dw_ek_borger, datotid_slut, diagnosegruppestreng FROM FOR_kohorte_indhold_pt_journal_psyk_somatik_inkl_2021_feb2022 WHERE datotid_slut IS NOT NULL"
        df = pl.from_pandas(sql_load(query, database="USR_PS_Forsk")).lazy()
        return self.format_diagnosis_columns(df=df)

    def format_diagnosis_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = (
            df.with_columns(
                pl.col("diagnosegruppestreng").str.split("#").alias("value"),
            )
            .drop("diagnosegruppestreng")
            .explode("value")
            .with_columns(
                [
                    pl.col("value").str.slice(offset=2).str.strip(),
                    pl.col("value").str.slice(offset=0, length=1).alias("type"),
                ],
            )
            .rename({"datotid_slut": "timestamp", "dw_ek_borger": "patient"})
        )

        return df


class MLMDataLoader(AbstractMLMDataLoader):
    @staticmethod
    def get_train_set(loaders: Sequence[EventDfLoader]) -> list[Patient]:
        event_data = pl.concat([loader.load_events() for loader in loaders])

        train_ids = pl.from_pandas(load_ids(split="train")).lazy()
        events_from_train = train_ids.join(event_data, on="dw_ek_borger", how="left")

        unpacked_patients = SourceEventDataframeUnpacker(
            column_names=PatientColumnNames(),
        ).unpack(source_event_dataframes=[events_from_train.collect()])

        return unpacked_patients
