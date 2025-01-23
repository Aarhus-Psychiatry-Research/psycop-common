# ruff: noqa
from psycop.common.feature_generation.loaders.raw.load_text import load_text_sfis
from psycop.common.model_training_v2.trainer.preprocessing.steps.row_filter_split import (
    RegionalFilter,
)

import polars as pl
from datetime import datetime

from psycop.projects.scz_bp.feature_generation.eligible_prediction_times.scz_bp_prediction_time_loader import (
    SczBpCohort,
)


def keyword_in_context(string: str) -> str:
    keyword_index = string.find("forklare")
    return string[keyword_index - 150 : keyword_index + 150]


def print_sample(df: pl.DataFrame):
    _sample = df.sample()
    text = _sample["value"].item()
    timestamp = _sample["timestamp"].item()
    outc_timestamp = _sample["outcome_timestamp"].item()
    note_type = _sample["overskrift"].item()

    print(f"Note type: {note_type}\nTime: {timestamp}\n Time of outcome: {outc_timestamp}")
    print(keyword_in_context(text))


if __name__ == "__main__":
    note_types = [
        "Observation af patient, Psykiatri",
        "Samtale med behandlingssigte",
        "Aktuelt psykisk",
        "Aktuelt socialt, Psykiatri",
        "Aftaler, Psykiatri",
        "Aktuelt somatisk, Psykiatri",
        "Objektivt psykisk",
        "Kontaktårsag",
        "Telefonnotat",
        "Semistruktureret diagnostisk interview",
        "Vurdering/konklusion",
    ]

    word_list = [" forklare "]

    text_df = pl.from_pandas(
        load_text_sfis(text_sfi_names=note_types, include_sfi_name=True, n_rows=None)
    )
    text_df = text_df.filter(
        pl.col("timestamp") < datetime(2016, 11, 1), pl.col("value").str.contains_any(word_list)
    )

    outcome_df = SczBpCohort().get_outcome_timestamps().frame
    outcome_df = outcome_df.rename({"timestamp": "outcome_timestamp", "value": "outc"})

    text_df = text_df.join(outcome_df, on="dw_ek_borger", how="left")

    only_outcome = text_df.filter(
        pl.col("outc") == 1, pl.col("timestamp") < pl.col("outcome_timestamp")
    )

    with pl.Config() as cfg:
        cfg.set_tbl_cols(-1)
        print(
            text_df["overskrift"]
            .value_counts()
            .with_columns((pl.col("count") / pl.col("count").sum() * 100).round(0).alias("procent"))
            .drop("count")
            .sort("procent", descending=True)
        )

    print_sample(only_outcome)
    print_sample(text_df)
