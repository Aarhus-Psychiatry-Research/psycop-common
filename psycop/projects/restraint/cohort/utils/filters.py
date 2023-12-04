from datetime import datetime

import pandas as pd
import polars as pl

from psycop.common.cohort_definition import PredictionTimeFilter
from psycop.projects.restraint.cohort.utils.config import (
    ADMISSION_TYPE,
    MIN_AGE,
    MIN_DATE,
    WASHOUT_INTERVAL_IN_DAYS,
)


class RestraintAdmissionTypeFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("pt_type") == ADMISSION_TYPE)


class RestraintMinDateFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("datotid_start") >= MIN_DATE)


class RestraintMinAgeFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("alder_start") >= MIN_AGE)


class RestraintCoercionTypeFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            (
                (pl.col("typetekst_sei") == "Bælte")
                & (pl.col("begrundtekst_sei") != "Frivillig bæltefiksering")
            )
            | (pl.col("typetekst_sei") == "Fastholden")
            | (pl.col("typetekst_sei") == "Beroligende medicin"),
        )


class RestraintWashoutFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            (
                pl.col("datotid_start") - pl.col("datotid_start_sei")
                >= pd.Timedelta(0, "days")
            )
            & (
                pl.col("datotid_start") - pl.col("datotid_start_sei")
                <= pd.Timedelta(WASHOUT_INTERVAL_IN_DAYS, "days")
            ),
        )


class RestraintWithinAdmissionsFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            (pl.col("datotid_start_sei") > pl.col("datotid_start"))
            & (pl.col("datotid_start_sei") < pl.col("datotid_slut")),
        )


class RestraintAdmissionFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            (pl.col("datotid_slut").is_not_null())
            & (pl.col("datotid_slut") <= datetime(year=2021, month=11, day=22)),
        )


class RestraintShakCodeFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            (pl.col("shakkode_ansvarlig") != "6600310")
            & (pl.col("shakkode_ansvarlig") != "6600021"),
        )


class RestraintTreatmentUnitFilter(PredictionTimeFilter):
    @staticmethod
    def apply(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("behandlingsomraade") != "Somatikken")
