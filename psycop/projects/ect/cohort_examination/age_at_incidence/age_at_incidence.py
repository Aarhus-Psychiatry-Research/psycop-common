from pathlib import Path
from typing import NewType

import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_demographic import birthdays
from psycop.common.global_utils.cache import shared_cache
from psycop.projects.ect.cohort_examination.incidence_by_time.model import incidence_by_time_model

AGE_COL_NAME = "age_in_years"


def age_at_incidence_facade(output_dir: Path):
    data = age_at_incidence_model()
    density_plot = age_incidence_distribution(model=data, density=False)
    cumulative_age_incidence(model=data)
    density_plot.save(
        output_dir / "age_incidence.png", limitsize=False, dpi=300, width=15, height=10
    )


def add_age(df: pl.DataFrame) -> pl.DataFrame:
    birthday_df = pl.from_pandas(birthdays())

    df = df.join(birthday_df, on="dw_ek_borger", how="inner")
    df = df.with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")).dt.total_days()).alias(AGE_COL_NAME)
    )
    df = df.with_columns((pl.col(AGE_COL_NAME) / 365.25).alias(AGE_COL_NAME))

    return df


AgeAtIncidenceModel = NewType("AgeAtIncidenceModel", pl.DataFrame)


@shared_cache().cache()
def age_at_incidence_model() -> AgeAtIncidenceModel:
    first_ect_with_age = add_age(incidence_by_time_model())
    return AgeAtIncidenceModel(first_ect_with_age)


def age_incidence_distribution(
    model: AgeAtIncidenceModel, limits: tuple[float, float] = (0, 100), density: bool = True
) -> pn.ggplot:
    p = (
        pn.ggplot(model, pn.aes(x=AGE_COL_NAME))
        + pn.scale_x_continuous(limits=limits)
        + pn.xlab("Age at first ECT")
        + (pn.geom_density() if density else pn.geom_histogram())
    )
    return p


def cumulative_age_incidence(model: AgeAtIncidenceModel) -> pn.ggplot:
    p = (
        pn.ggplot(model, pn.aes(x=AGE_COL_NAME))
        + pn.labs(x="Age at first ECT", y="Cumulative incidence")
        + pn.stat_ecdf()
    )
    return p
