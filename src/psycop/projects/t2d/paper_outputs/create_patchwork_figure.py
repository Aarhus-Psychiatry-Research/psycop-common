from datetime import datetime
from typing import Callable, Sequence

from psycop.common.model_evaluation.patchwork.patchwork_grid import (
    create_patchwork_grid,
)
from psycop.projects.t2d.paper_outputs.config import FIGURES_PATH
from psycop.projects.t2d.utils.best_runs import ModelRun
from wasabi import Printer

msg = Printer(timestamp=True)


def t2d_create_patchwork_figure(
    run: ModelRun, plot_fns: Sequence[Callable], output_filename: str
):
    """Create a patchwork figure from plot_fns. All plot_fns must only need a ModelRun object as input, and return a plotnine ggplot output."""
    for output_str, value in (
        ("Run group", run.group.name),
        ("Model_type", run.model_type),
        ("Lookahead days", run.cfg.preprocessing.pre_split.min_lookahead_days),
    ):
        msg.info(f"    {output_str}: {value}")

    plots = []

    any_plot_failed = False

    for fn in plot_fns:
        try:
            now = datetime.now()
            p = fn(run)
            plots.append(p)
            p.save(FIGURES_PATH / f"{fn.__name__}.png")
            finished = datetime.now()

            msg.good(
                f"{fn.__name__} finished in {round((finished - now).seconds, 0)} seconds",
            )
        except Exception:
            msg.fail(f"{fn.__name__} failed")
            any_plot_failed = True
    datetime.now()

    if not any_plot_failed:
        grid = create_patchwork_grid(
            plots=plots,
            single_plot_dimensions=(5, 5),
            n_in_row=2,
        )
        grid.savefig(FIGURES_PATH / output_filename)
