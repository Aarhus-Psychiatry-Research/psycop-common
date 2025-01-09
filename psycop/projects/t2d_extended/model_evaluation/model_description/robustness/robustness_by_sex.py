import plotnine as pn

from psycop.common.model_evaluation.binary.subgroup_data import get_auroc_by_input_df
from psycop.projects.t2d_extended.model_evaluation.model_description.robustness.robustness_plot import (
    t2d_extended_plot_robustness,
)
from psycop.projects.t2d_extended.model_evaluation.selected_runs import get_best_eval_pipeline
from psycop.projects.t2d_extended.utils.pipeline_objects import T2DExtendedPipelineRun


def t2d_extended_auroc_by_sex(run: T2DExtendedPipelineRun) -> pn.ggplot:
    eval_ds = run.pipeline_outputs.get_eval_dataset(custom_columns=["is_female"])

    df = get_auroc_by_input_df(
        eval_dataset=eval_ds,
        input_values=eval_ds.is_female,  # type: ignore
        input_name="is_female",
        confidence_interval=True,
        bin_continuous_input=False,
    )

    int_to_sex = {0: "Male", 1: "Female"}
    df["Sex"] = df["is_female"].map(int_to_sex).astype("category")

    return t2d_extended_plot_robustness(df, x_column="Sex", line_y_col_name="auroc", xlab="Sex")


if __name__ == "__main__":
    t2d_extended_auroc_by_sex(run=get_best_eval_pipeline())
