import plotnine as pn
from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.t2d.paper_outputs.config import BEST_EVAL_PIPELINE
from psycop.projects.t2d.paper_outputs.model_description.robustness.robustness_plot import (
    t2d_plot_robustness,
)
from psycop.projects.t2d.utils.best_runs import PipelineRun


def t2d_auroc_by_sex(run: PipelineRun) -> pn.ggplot:
    print("Plotting AUC by sex")
    eval_ds = run.get_eval_dataset(custom_columns=["is_female"])

    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.is_female,  # type: ignore
        input_name="is_female",
        confidence_interval=True,
        bin_continuous_input=False,
    )

    int_to_sex = {0: "Male", 1: "Female"}
    df["Sex"] = df["is_female"].map(int_to_sex).astype("category")

    return t2d_plot_robustness(
        df,
        x_column="Sex",
        line_y_col_name="auroc",
        xlab="Sex",
        figure_file_name="t2d_auroc_by_sex",
    )


if __name__ == "__main__":
    t2d_auroc_by_sex(run=BEST_EVAL_PIPELINE)
