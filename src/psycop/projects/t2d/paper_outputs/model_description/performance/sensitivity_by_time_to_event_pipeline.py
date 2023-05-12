from psycop.common.model_evaluation.binary.time.timedelta_plots import (
    plot_sensitivity_by_time_to_event,
)
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN, FIGURES_PATH


def incidence_by_time_until_outcome_pipeline():
    eval_ds = EVAL_RUN.get_eval_dataset()

    plot_sensitivity_by_time_to_event(
        eval_dataset=eval_ds,
        bins=range(0, 37, 6),
        positive_rates=[0.05, 0.03, 0.01],
        bin_unit="M",
        save_path=FIGURES_PATH / "sensitivity_by_time_to_event.png",
    )


if __name__ == "__main__":
    incidence_by_time_until_outcome_pipeline()
