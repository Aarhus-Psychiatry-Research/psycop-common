import plotnine as pn
from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.t2d.paper_outputs.config import BEST_EVAL_PIPELINE
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)
from psycop.projects.t2d.utils.best_runs import PipelineRun


def t2d_auroc_by_age(run: PipelineRun) -> pn.ggplot:
    print("Plotting AUROC by age")
    eval_ds = run.get_eval_dataset()

    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.age,  # type: ignore
        input_name="age",
        bins=[18, *range(20, 80, 10)],
        bin_continuous_input=True,
        confidence_interval=True,
    )

    p = t2d_plot_robustness(
        df,
        x_column="age_binned",
        line_y_col_name="auroc",
        xlab="Age",
        figure_file_name="auroc_by_age.png",
        rotate_x_axis_labels_degrees=45,
    )

    return p


if __name__ == "__main__":
    t2d_auroc_by_age(run=BEST_EVAL_PIPELINE)
