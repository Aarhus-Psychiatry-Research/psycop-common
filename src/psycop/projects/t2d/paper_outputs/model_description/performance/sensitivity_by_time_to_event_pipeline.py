from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, FIGURES_PATH
from psycop.projects.t2d.utils.best_runs import Run


def sensitivity_by_time_to_event(run: Run):
    run.get_eval_dataset()

    plot_path = FIGURES_PATH / "sensitivity_by_time_to_event.png"
    # TODO: Create plot
    print(f"Saving sensitivity by time to event plot to {plot_path}")


if __name__ == "__main__":
    sensitivity_by_time_to_event(run=EVAL_RUN)
