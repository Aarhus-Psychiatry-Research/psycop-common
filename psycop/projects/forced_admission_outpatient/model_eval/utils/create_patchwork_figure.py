from collections.abc import Sequence
from datetime import datetime
from typing import Callable

from wasabi import Printer

from psycop.common.model_evaluation.patchwork.patchwork_grid import (
    create_patchwork_grid,
)
from psycop.projects.forced_admission_outpatient.utils.pipeline_objects import (
    ForcedAdmissionOutpatientPipelineRun,
)

msg = Printer(timestamp=True)


def fa_outpatient_create_patchwork_figure(
    run: ForcedAdmissionOutpatientPipelineRun,
    plot_fns: Sequence[Callable],  # type: ignore
    output_filename: str,
    single_plot_dimensions: tuple[float, float] = (5, 5),
    n_in_row: int = 2,
):
    """Create a patchwork figure from plot_fns. All plot_fns must only need a ModelRun object as input, and return a plotnine ggplot output."""
    for output_str, value in (
        ("Run group", run.group.group_name),
        ("Model_type", run.model_type),
        ("Lookahead days", run.inputs.cfg.preprocessing.pre_split.min_lookahead_days),
    ):
        msg.info(f"    {output_str}: {value}")

    plots = []

    any_plot_failed = False

    for fn in plot_fns:
        output_path = run.paper_outputs.paths.figures / f"{fn.__name__}.png"
        try:
            now = datetime.now()
            p = fn(run)
            plots.append(p)
            p.save(output_path)
            finished = datetime.now()

            msg.good(f"{fn.__name__} finished in {round((finished - now).seconds, 0)} seconds")
        except Exception:
            msg.fail(f"{fn.__name__} failed")
            any_plot_failed = True
    datetime.now()

    if not any_plot_failed:
        grid = create_patchwork_grid(
            plots=plots, single_plot_dimensions=single_plot_dimensions, n_in_row=n_in_row
        )
        grid_output_path = run.paper_outputs.paths.figures / output_filename
        grid.savefig(grid_output_path)
        msg.good(f"Saved figure to {grid_output_path}")
