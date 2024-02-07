import pandas as pd

from psycop.common.model_evaluation.patchwork.patchwork_grid import create_patchwork_grid
from psycop.common.model_evaluation.utils import TEST_PLOT_PATH
from psycop.common.test_utils.str_to_df import str_to_df
from psycop.projects.t2d.paper_outputs.model_description.performance.sensitivity_by_time_to_event_pipeline import (
    _plot_sensitivity_by_time_to_event,  # type: ignore
)


def test_plot_sensitivity_by_time_to_event_with_patchwork():
    df = str_to_df(
        """unit_from_event_binned,sensitivity,actual_positive_rate,ci_lower,ci_upper,
        1-3,0.1,A,0.05,0.15,
        4-6,0.2,A,0.15,0.25,
        7-10,0.3,A,0.25,0.35,
        1-3,0.2,B,0.15,0.25,
        4-6,0.3,B,0.25,0.35,
        7-10,0.4,B,0.35,0.45,
        """
    )

    df["unit_from_event_binned"] = pd.Categorical(df["unit_from_event_binned"], ordered=True)

    plots = [_plot_sensitivity_by_time_to_event(df) for _ in range(3)]

    grid = create_patchwork_grid(plots=plots, single_plot_dimensions=(5, 5), n_in_row=2)
    grid.savefig(TEST_PLOT_PATH / "sensitivity_by_time_to_event.png")
