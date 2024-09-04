from pathlib import Path

from psycop.projects.ect.cohort_examination.incidence_by_time.model import incidence_by_time_model
from psycop.projects.ect.cohort_examination.incidence_by_time.view import (
    incidence_by_time_distribution_density,
    incidence_by_time_distribution_histogram,
    incidence_by_time_faceted,
)


def incidence_by_time_facade(output_dir: Path):
    data = incidence_by_time_model()
    distribution_plot_density = incidence_by_time_distribution_density(data)
    distribution_plot_histogram = incidence_by_time_distribution_histogram(data)
    distribution_plot_density.save(
        output_dir / "incidence_by_time.png", limitsize=False, dpi=300, width=15, height=10
    )

    facet_plot = incidence_by_time_faceted(data)
    facet_plot.save(
        output_dir / "incidence_by_time_faceted.png", limitsize=False, dpi=300, width=15, height=10
    )
