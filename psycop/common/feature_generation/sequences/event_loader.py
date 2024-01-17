from abc import abstractmethod
from typing import Protocol, runtime_checkable

import polars as pl

from ...sequence_models.registry import SequenceRegistry
from ..loaders.raw.sql_load import sql_load


@runtime_checkable
class EventLoader(Protocol):
    @abstractmethod
    def load_events(self) -> pl.LazyFrame:
        ...


@SequenceRegistry.event_loaders.register("diagnoses")
class DiagnosisLoader(EventLoader):
    """Load all diagnoses for all patients."""

    def __init__(self) -> None:
        super().__init__()

    def load_events(self) -> pl.LazyFrame:
        """Load all a-diagnoses"""
        query = "SELECT dw_ek_borger, datotid_slut, diagnosegruppestreng FROM [fct].[FOR_kohorte_indhold_pt_journal_psyk_somatik_inkl_2021_feb2022] WHERE datotid_slut IS NOT NULL"
        df = pl.from_pandas(sql_load(query, database="USR_PS_Forsk")).lazy()
        return self.preprocess_diagnosis_columns(df=df)

    def preprocess_diagnosis_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
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
                    pl.col("value").str.split_exact(
                        ":",
                        1,
                    ),  # Split diagnosis column into prefix (e.g., A, B, or Z) and diagnosis code
                    pl.lit("diagnosis").alias(
                        "source",
                    ),  # Add a source column, indicating diagnoses
                ],
            )
            .unnest("value")  # Unnest prefix and diagnosis code into separate columns
            .with_columns(
                pl.col("field_1")
                .str.replace(
                    "^D",
                    "",
                )
                .str.strip(),
            )  # In the diagnosis DF432, the D is only in the Danish system and doesn't carry meaning. Remove it.)
            .rename(
                {"datotid_slut": "timestamp", "field_0": "type", "field_1": "value"},
            )
            .filter(
                pl.col("type").str.lengths() < 2,
            )  # Removes HCO "diagnosis metadata", which is of too poor quality.
        ).unique(maintain_order=True)

        return df.drop_nulls()  # Drop all rows with no diagnosis
