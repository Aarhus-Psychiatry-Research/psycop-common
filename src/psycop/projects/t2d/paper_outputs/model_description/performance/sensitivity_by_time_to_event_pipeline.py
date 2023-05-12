from psycop.common.model_evaluation.binary.time.timedelta_plots import (
    plot_sensitivity_by_time_to_event,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, FIGURES_PATH
from psycop.projects.t2d.utils.best_runs import Run


def sensitivity_by_time_to_event(run: Run):
    eval_ds = run.get_eval_dataset()

    plot_path = FIGURES_PATH / "sensitivity_by_time_to_event.png"
    plot_sensitivity_by_time_to_event(
        eval_dataset=eval_ds,
        bins=range(0, 37, 6),
        positive_rates=[0.05, 0.03, 0.01],
        bin_unit="M",
        save_path=plot_path,
    )
    print(f"Saving sensitivity by time to event plot to {plot_path}")


if __name__ == "__main__":
    sensitivity_by_time_to_event(run=EVAL_RUN)
