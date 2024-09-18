from pathlib import Path

from psycop.projects.t2d_bigdata.cohort_examination.incidence_by_time.model import incidence_by_time_model
from psycop.projects.t2d_bigdata.cohort_examination.incidence_by_time.view import (
    incidence_by_time_distribution,
    incidence_by_time_faceted,
)


def incidence_by_time_facade(output_dir: Path):
    data = incidence_by_time_model()
    distribution_plot = incidence_by_time_distribution(data)
    distribution_plot.save(
        output_dir / "incidence_by_time.png", limitsize=False, dpi=300, width=15, height=10
    )

    facet_plot = incidence_by_time_faceted(data)
    facet_plot.save(
        output_dir / "incidence_by_time_faceted.png", limitsize=False, dpi=300, width=15, height=10
    )
