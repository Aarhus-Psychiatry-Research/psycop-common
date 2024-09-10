# remove admission after ect
# identify closest admission
# calculate timedelta
from pathlib import Path
from typing import NewType

import plotnine as pn
import polars as pl

from psycop.common.feature_generation.loaders.raw.load_visits import admissions
from psycop.common.global_utils.cache import shared_cache
from psycop.projects.ect.feature_generation.cohort_definition.outcome_specification.combined import (
    get_first_ect_indicator,
)

TimeFromAdmissionModel = NewType("TimeFromAdmissionModel", pl.DataFrame)


def time_from_admission_to_incidence_facade(output_dir: Path):
    data = time_from_admission_model()

    p = time_from_admission_view(model=data)
    p.save(
        output_dir / "incidence_by_time_faceted.png", limitsize=False, dpi=300, width=15, height=10
    )


@shared_cache().cache()
def time_from_admission_model() -> TimeFromAdmissionModel:
    first_ect = pl.from_pandas(get_first_ect_indicator()).rename({"timestamp": "timestamp_ect"})
    admissions_df = (
        pl.from_pandas(admissions()).rename({"timestamp": "timestamp_admission"}).drop("value")
    )

    # some patients get ECT before an admission, losing ~700
    closest_admission = (
        first_ect.join(admissions_df, how="left", on="dw_ek_borger")
        .filter(pl.col("timestamp_ect") > pl.col("timestamp_admission"))
        .with_columns((pl.col("timestamp_ect") - pl.col("timestamp_admission")).alias("timedelta"))
        .sort(["dw_ek_borger", "timedelta"], descending=False)
        .group_by("dw_ek_borger")
        .first()
    )
    return TimeFromAdmissionModel(closest_admission)


def time_from_admission_view(
    model: TimeFromAdmissionModel, limits: tuple[float, float] = (0, 60)
) -> pn.ggplot:
    p = (
        pn.ggplot(model.with_columns(pl.col("timedelta").dt.days()), pn.aes(x="timedelta"))
        + pn.geom_histogram(bins=30)
        + pn.geom_vline(xintercept=7)
        + pn.scale_x_continuous(limits=limits)
        + pn.labs(x="Days from closest admission")
    )

    return p
