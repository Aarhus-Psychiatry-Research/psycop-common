import plotnine as pn
from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.t2d.paper_outputs.config import EVAL_RUN
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    plot_robustness,
)
from psycop.projects.t2d.utils.best_runs import ModelRun


def auroc_by_age(run: ModelRun) -> pn.ggplot:
    print("Plotting AUC by age")
    eval_ds = run.get_eval_dataset()

    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.age,  # type: ignore
        input_name="age",
        bins=[18, *range(20, 80, 10)],
        bin_continuous_input=True,
        confidence_interval=True,
    )

    return plot_robustness(
        df,
        x_column="age_binned",
        line_y_col_name="auroc",
        bar_y_col_name="proportion_of_n",
        xlab="Age",
        ylab="AUROC / Proportion of patients",
    )


if __name__ == "__main__":
    auroc_by_age(run=EVAL_RUN)
